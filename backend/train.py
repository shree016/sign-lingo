# backend/train.py
#
# Loads the landmark CSV, prepares the data, trains the model,
# and saves the trained weights to the model/ folder.
#
# Run with: python backend/train.py
#
# Expected training time: 3–8 minutes on a modern CPU

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras

# Must come before any 'from backend...' imports
# Adds the project root (sign-lingo/) to Python's module search path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.model import build_model

# ── CONFIG ─────────────────────────────────────────────────────────────────────
CSV_PATH   = "data/landmarks/landmarks.csv"
MODEL_SAVE = "model/sign_model.keras"
PLOT_SAVE  = "model/training_history.png"

EPOCHS      = 50    # Max training epochs (early stopping may halt sooner)
BATCH_SIZE  = 64    # Samples processed before each weight update
VALIDATION  = 0.15  # 15% of data for validation during training
TEST_SPLIT  = 0.15  # 15% of data held back for final evaluation
RANDOM_SEED = 42
# ──────────────────────────────────────────────────────────────────────────────


def load_and_prepare_data():
    """
    Loads the CSV, splits features from labels, encodes labels
    as integers, and splits into train / val / test sets.

    Returns:
        X_train, X_val, X_test : numpy arrays of shape (n, 63)
        y_train, y_val, y_test : numpy arrays of integer labels
        encoder                : fitted LabelEncoder instance
    """
    print("Loading data...")
    df = pd.read_csv(CSV_PATH)
    print(f"  Loaded {len(df):,} samples, {df.shape[1]} columns")

    # All columns except the last are the 63 landmark coordinates
    X = df.iloc[:, :-1].values   # Shape: (61806, 63)
    y = df.iloc[:, -1].values    # Shape: (61806,) — string labels e.g. 'A'

    # Convert string labels to integers
    # LabelEncoder sorts alphabetically: A=0, B=1, ..., Z=25, space=26
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    print(f"  Classes ({len(encoder.classes_)}): {list(encoder.classes_)}")

    # Split 1: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded,
        test_size=(VALIDATION + TEST_SPLIT),  # 0.30
        random_state=RANDOM_SEED,
        stratify=y_encoded  # Keep class proportions equal across splits
    )

    # Split 2: 30% temp → 15% val + 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=y_temp
    )

    print(f"\n  Train:      {len(X_train):,} samples (70%)")
    print(f"  Validation: {len(X_val):,} samples (15%)")
    print(f"  Test:       {len(X_test):,} samples (15%)")

    return X_train, X_val, X_test, y_train, y_val, y_test, encoder


def get_callbacks():
    """
    Three callbacks that control training automatically:

    1. ModelCheckpoint  — saves model whenever val_accuracy improves
    2. EarlyStopping    — stops training if no improvement for 10 epochs
    3. ReduceLROnPlateau — halves learning rate when progress stalls
    """
    os.makedirs("model", exist_ok=True)

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=MODEL_SAVE,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,               # Stop after 10 epochs with no improvement
        restore_best_weights=True, # Reload best weights when stopping
        verbose=1
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,    # Multiply LR by 0.5 when triggered
        patience=5,    # Trigger after 5 epochs with no improvement
        min_lr=1e-6,   # Never drop below this
        verbose=1
    )

    return [checkpoint, early_stop, reduce_lr]


def plot_training_history(history):
    """
    Plots accuracy and loss curves for training and validation sets.

    What healthy curves look like:
    - Both train and val accuracy rise together toward 90%+
    - Both train and val loss fall together
    - If train >> val accuracy: overfitting (model memorised training data)
    - If both are low: underfitting (need more epochs or bigger model)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Sign Lingo — Training History", fontsize=14, fontweight='bold')

    epochs_ran = range(1, len(history.history['accuracy']) + 1)

    # Accuracy
    ax1.plot(epochs_ran, history.history['accuracy'],
             'b-', label='Training', linewidth=2)
    ax1.plot(epochs_ran, history.history['val_accuracy'],
             'r--', label='Validation', linewidth=2)
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

    # Loss
    ax2.plot(epochs_ran, history.history['loss'],
             'b-', label='Training', linewidth=2)
    ax2.plot(epochs_ran, history.history['val_loss'],
             'r--', label='Validation', linewidth=2)
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_SAVE, dpi=100, bbox_inches='tight')
    plt.show()
    print(f"✅ Training plot saved to {PLOT_SAVE}")


def train():
    """Main training function — ties everything together."""

    print("=" * 55)
    print("  Sign Lingo — Model Training")
    print("=" * 55)

    # Step 1 — Load and prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, encoder = \
        load_and_prepare_data()

    # Step 2 — Save class order so app.py can decode predictions later
    # e.g. model outputs 0 → look up classes.npy[0] → 'A'
    os.makedirs("model", exist_ok=True)
    np.save("model/classes.npy", encoder.classes_)
    print(f"\n  Class order saved to model/classes.npy")

    # Step 3 — Build model
    num_classes = len(encoder.classes_)
    print(f"\nBuilding model for {num_classes} classes...")
    model = build_model(num_classes=num_classes)
    model.summary()

    # Step 4 — Train
    print(f"\nTraining for up to {EPOCHS} epochs "
          f"(early stopping if no improvement for 10 epochs)...\n")

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=get_callbacks(),
        verbose=1
    )

    # Step 5 — Final evaluation on unseen test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    print(f"\n{'='*55}")
    print(f"  Final Test Accuracy : {test_accuracy * 100:.2f}%")
    print(f"  Final Test Loss     : {test_loss:.4f}")
    print(f"  Model saved to      : {MODEL_SAVE}")
    print(f"  Classes saved to    : model/classes.npy")
    print(f"{'='*55}")

    # Step 6 — Plot training curves
    plot_training_history(history)

    return model, history, encoder


if __name__ == "__main__":
    train()