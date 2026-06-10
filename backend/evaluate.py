# backend/evaluate.py
#
# Loads the trained model and evaluates it thoroughly:
#   - Confusion matrix (which signs get confused with each other)
#   - Per-class precision, recall, F1 score
#   - Top confusions (the most common mistakes)
#   - Confidence distribution (how sure the model is when predicting)
#
# Run with: python backend/evaluate.py

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── CONFIG ─────────────────────────────────────────────────────────────────────
CSV_PATH    = "data/landmarks/landmarks.csv"
MODEL_PATH  = "model/sign_model.keras"
CLASSES_PATH = "model/classes.npy"

RANDOM_SEED = 42
TEST_SPLIT  = 0.15
VALIDATION  = 0.15
# ──────────────────────────────────────────────────────────────────────────────


def load_test_data():
    """
    Recreates the exact same test split used during training.
    We use the same RANDOM_SEED so the split is identical —
    this ensures we're evaluating on truly unseen data.
    """
    print("Loading data...")
    df = pd.read_csv(CSV_PATH)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Reproduce the exact same splits from train.py
    _, X_temp, _, y_temp = train_test_split(
        X, y_encoded,
        test_size=(VALIDATION + TEST_SPLIT),
        random_state=RANDOM_SEED,
        stratify=y_encoded
    )
    _, X_test, _, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=y_temp
    )

    print(f"  Test samples: {len(X_test):,}")
    return X_test, y_test, encoder


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plots a heatmap of the confusion matrix.

    Each cell (row i, col j) shows how many times
    the true class i was predicted as class j.

    The diagonal = correct predictions (we want these bright).
    Off-diagonal = mistakes (we want these dark/empty).
    """
    cm = confusion_matrix(y_true, y_pred)

    # Normalize to percentages (easier to read than raw counts)
    # Each row sums to 1.0 — shows the % of each true class
    # that was predicted as each other class
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    fig.suptitle("Sign Lingo — Confusion Matrix", fontsize=15, fontweight='bold')

    # ── Raw counts ─────────────────────────────────────────────────────────────
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',              # Integer format
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
        linewidths=0.5,
        cbar=False
    )
    axes[0].set_title('Raw Counts', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=11)
    axes[0].set_ylabel('True Label', fontsize=11)
    axes[0].tick_params(axis='both', labelsize=9)

    # ── Normalized percentages ─────────────────────────────────────────────────
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',           # 2 decimal places
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
        linewidths=0.5,
        vmin=0, vmax=1       # Fix scale 0→1 so diagonal always looks bright
    )
    axes[1].set_title('Normalized (% per true class)', fontsize=12)
    axes[1].set_xlabel('Predicted Label', fontsize=11)
    axes[1].set_ylabel('True Label', fontsize=11)
    axes[1].tick_params(axis='both', labelsize=9)

    plt.tight_layout()
    plt.savefig("model/confusion_matrix.png", dpi=120, bbox_inches='tight')
    plt.show()
    print("✅ Confusion matrix saved to model/confusion_matrix.png")

    return cm


def print_classification_report(y_true, y_pred, class_names):
    """
    Prints per-class precision, recall and F1 score.

    Good output looks like this for each class:
        precision  recall  f1-score  support
    A     0.99      1.00    1.00      450
    B     0.98      0.99    0.99      448

    support = number of test samples for that class
    """
    print("\n" + "="*65)
    print("  Per-Class Classification Report")
    print("="*65)
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4    # 4 decimal places
    )
    print(report)

    # Save to file too
    with open("model/classification_report.txt", "w") as f:
        f.write("Sign Lingo — Classification Report\n")
        f.write("="*65 + "\n")
        f.write(report)
    print("✅ Report saved to model/classification_report.txt")


def print_top_confusions(cm, class_names, top_n=10):
    """
    Finds and prints the most common mistakes the model makes.
    These are the off-diagonal cells with the highest values.

    This is the most actionable insight — it tells you exactly
    which sign pairs are most often mixed up.
    """
    print(f"\n{'='*65}")
    print(f"  Top {top_n} Most Confused Sign Pairs")
    print(f"{'='*65}")
    print(f"  {'True Sign':<12} {'Predicted As':<14} {'Times Wrong':>12}")
    print(f"  {'-'*40}")

    # Find all off-diagonal (wrong) predictions
    confusions = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confusions.append((class_names[i], class_names[j], cm[i, j]))

    # Sort by most confused first
    confusions.sort(key=lambda x: x[2], reverse=True)

    for true_label, pred_label, count in confusions[:top_n]:
        print(f"  {true_label:<12} → {pred_label:<14} {count:>8} times")

    if not confusions:
        print("  🎉 No confusions found — perfect classification!")


def plot_per_class_accuracy(y_true, y_pred, class_names):
    """
    Bar chart showing accuracy for each individual sign.
    Makes it easy to spot any underperforming classes at a glance.
    """
    cm = confusion_matrix(y_true, y_pred)

    # Per-class accuracy = diagonal / row sum
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(
        class_names,
        per_class_acc * 100,
        color=['#e74c3c' if acc < 0.95 else '#2ecc71' for acc in per_class_acc],
        edgecolor='white',
        linewidth=0.5
    )

    # Add value labels on top of each bar
    for bar, acc in zip(bars, per_class_acc):
        if acc < 0.995:   # Only label bars that aren't near 100%
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f'{acc*100:.1f}%',
                ha='center', va='bottom', fontsize=8
            )

    ax.set_title('Per-Class Accuracy — Sign Lingo', fontsize=13, fontweight='bold')
    ax.set_xlabel('Sign')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim([90, 101])   # Zoom in — all should be near 100%
    ax.axhline(y=99, color='orange', linestyle='--',
               linewidth=1, label='99% threshold')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("model/per_class_accuracy.png", dpi=100, bbox_inches='tight')
    plt.show()
    print("✅ Per-class accuracy chart saved to model/per_class_accuracy.png")


def plot_confidence_distribution(y_true, y_pred_probs, y_pred, class_names):
    """
    Shows how confident the model is when it makes correct vs incorrect
    predictions. A well-calibrated model should be:
    - Very confident (>0.90) on correct predictions
    - Less confident on wrong predictions

    If the model is highly confident on wrong predictions,
    that's a sign of overconfidence — a potential problem.
    """
    # Get the max probability for each prediction
    # This is how confident the model is in its top choice
    max_probs = np.max(y_pred_probs, axis=1)

    correct_mask   = (y_pred == y_true)
    incorrect_mask = ~correct_mask

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        max_probs[correct_mask],
        bins=50, alpha=0.7,
        color='#2ecc71', label=f'Correct ({correct_mask.sum():,})',
        density=True
    )

    if incorrect_mask.sum() > 0:
        ax.hist(
            max_probs[incorrect_mask],
            bins=50, alpha=0.7,
            color='#e74c3c', label=f'Incorrect ({incorrect_mask.sum():,})',
            density=True
        )

    ax.set_title('Prediction Confidence Distribution', fontsize=13,
                 fontweight='bold')
    ax.set_xlabel('Model Confidence (max softmax probability)')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("model/confidence_distribution.png", dpi=100, bbox_inches='tight')
    plt.show()
    print("✅ Confidence chart saved to model/confidence_distribution.png")


def test_single_sample(model, X_test, y_test, encoder, num_samples=5):
    """
    Picks random samples from the test set and shows the model's
    top 3 predictions with confidence scores for each.

    This simulates what happens during real-time detection —
    one set of 63 landmark values → ranked list of predictions.
    """
    print(f"\n{'='*65}")
    print(f"  Sample Predictions (random test samples)")
    print(f"{'='*65}")

    # Get raw logits from the model
    logits = model.predict(X_test, verbose=0)

    # Apply softmax manually to convert logits → probabilities
    # axis=1 means softmax across classes for each sample
    probs = tf.nn.softmax(logits, axis=1).numpy()

    indices = np.random.choice(len(X_test), num_samples, replace=False)

    for idx in indices:
        true_label = encoder.classes_[y_test[idx]]
        sample_probs = probs[idx]

        # Get top 3 predictions sorted by confidence
        top3_idx  = np.argsort(sample_probs)[::-1][:3]
        top3_conf = sample_probs[top3_idx]
        top3_cls  = encoder.classes_[top3_idx]

        correct = "✅" if top3_cls[0] == true_label else "❌"
        print(f"\n  True label: '{true_label}'  {correct}")
        for cls, conf in zip(top3_cls, top3_conf):
            marker = "←" if cls == true_label else "  "
            print(f"    {cls}: {conf*100:6.2f}%  {marker}")


def evaluate():
    """Main evaluation function."""

    print("=" * 55)
    print("  Sign Lingo — Model Evaluation")
    print("=" * 55)

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"\nLoading model from {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH)
    print("  ✅ Model loaded")

    # ── Load class names ───────────────────────────────────────────────────────
    class_names = np.load(CLASSES_PATH, allow_pickle=True)
    print(f"  Classes: {list(class_names)}")

    # ── Load test data ─────────────────────────────────────────────────────────
    X_test, y_test, encoder = load_test_data()

    # ── Get predictions ────────────────────────────────────────────────────────
    print("\nRunning predictions on test set...")
    logits = model.predict(X_test, verbose=1)

    # Convert logits → probabilities
    y_pred_probs = tf.nn.softmax(logits, axis=1).numpy()

    # Get the index of the highest probability = predicted class
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Overall accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"\n  Overall Accuracy: {accuracy*100:.4f}%")

    # ── Run all evaluations ────────────────────────────────────────────────────
    # 1. Classification report (precision, recall, F1 per class)
    print_classification_report(y_test, y_pred, class_names)

    # 2. Confusion matrix heatmap
    cm = plot_confusion_matrix(y_test, y_pred, class_names)

    # 3. Top confused pairs
    print_top_confusions(cm, class_names, top_n=10)

    # 4. Per-class accuracy bar chart
    plot_per_class_accuracy(y_test, y_pred, class_names)

    # 5. Confidence distribution
    plot_confidence_distribution(y_test, y_pred_probs, y_pred, class_names)

    # 6. Sample predictions
    test_single_sample(model, X_test, y_test, encoder)

    print(f"\n{'='*55}")
    print(f"  Evaluation complete!")
    print(f"  All charts saved to model/")
    print(f"{'='*55}")


if __name__ == "__main__":
    evaluate()