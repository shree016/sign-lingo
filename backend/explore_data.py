# backend/explore_data.py
#
# This script answers 4 questions about our dataset:
# 1. How many classes (letters) do we have?
# 2. How many images per class?
# 3. What do the images look like (size, color mode)?
# 4. Are there any problems (missing data, imbalance)?
#
# Run it with: python backend/explore_data.py

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


# --- CONFIGURATION ---
# This is the only line you might need to change if your
# folder ended up with a different name after unzipping
DATA_DIR = "data/raw/asl_alphabet_train/asl_alphabet_train"


def explore_dataset(data_dir):
    """
    Walks through the dataset folder and prints a full report.
    """

    # --- Check the folder exists ---
    if not os.path.exists(data_dir):
        print(f"❌ Folder not found: {data_dir}")
        print("Check your DATA_DIR path above — the folder name after")
        print("unzipping might be slightly different.")
        print("\nFolders inside data/raw/:")
        for item in os.listdir("data/raw"):
            print(f"  {item}")
        return

    # --- Count classes and images ---
    # os.listdir gives us all items in the folder
    # We filter to only directories (each dir = one letter class)
    classes = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    print(f"✅ Dataset found at: {data_dir}")
    print(f"📁 Total classes: {len(classes)}")
    print(f"📋 Classes: {classes}\n")

    # --- Count images in each class ---
    class_counts = {}
    total_images = 0

    for class_name in classes:
        class_path = os.path.join(data_dir, class_name)
        # Count only image files (not hidden files like .DS_Store)
        images = [
            f for f in os.listdir(class_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        class_counts[class_name] = len(images)
        total_images += len(images)

    print(f"🖼️  Total images: {total_images:,}")
    print(f"📊 Images per class:\n")

    # Print a simple text bar chart
    for class_name, count in class_counts.items():
        bar = "█" * (count // 100)   # Each block = 100 images
        print(f"  {class_name:10} {count:5,}  {bar}")

    # --- Inspect a sample image ---
    print("\n🔍 Inspecting a sample image...")
    sample_class = classes[0]
    sample_path = os.path.join(data_dir, sample_class)
    sample_file = os.listdir(sample_path)[0]
    sample_img_path = os.path.join(sample_path, sample_file)

    img = cv2.imread(sample_img_path)
    # OpenCV loads images in BGR format, not RGB
    # For display, we convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(f"  File: {sample_file}")
    print(f"  Shape: {img.shape}")        # (height, width, channels)
    print(f"  Height: {img.shape[0]}px")
    print(f"  Width:  {img.shape[1]}px")
    print(f"  Color channels: {img.shape[2]} (BGR)")
    print(f"  Pixel value range: {img.min()} – {img.max()}")

    # --- Check for class imbalance ---
    counts = list(class_counts.values())
    min_count = min(counts)
    max_count = max(counts)
    print(f"\n⚖️  Class balance check:")
    print(f"  Min images in a class: {min_count}")
    print(f"  Max images in a class: {max_count}")
    if max_count - min_count < 500:
        print("  ✅ Dataset is well-balanced!")
    else:
        print("  ⚠️  Some imbalance detected — we'll handle this during training")

    # --- Visualize sample images from each class ---
    print("\n📸 Generating sample image grid...")
    plot_samples(data_dir, classes)


def plot_samples(data_dir, classes):
    """
    Shows one sample image from each class in a grid.
    This helps you visually confirm the dataset loaded correctly.
    """
    # We'll show only A-Z (26 letters), skip SPACE/DELETE/NOTHING
    # for a cleaner grid
    letter_classes = [c for c in classes if len(c) == 1]

    # Create a 4-row × 7-column grid (28 slots for 26 letters + 2 empty)
    fig, axes = plt.subplots(4, 7, figsize=(14, 8))
    fig.suptitle("Sign Lingo — ASL Dataset Sample Images",
                 fontsize=16, fontweight='bold')

    for idx, ax in enumerate(axes.flatten()):
        if idx < len(letter_classes):
            class_name = letter_classes[idx]
            class_path = os.path.join(data_dir, class_name)
            img_file = os.listdir(class_path)[0]   # First image in folder
            img_path = os.path.join(class_path, img_file)

            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            ax.imshow(img_rgb)
            ax.set_title(class_name, fontsize=12, fontweight='bold')
            ax.axis('off')   # Hide axis ticks
        else:
            ax.axis('off')   # Hide empty grid slots

    plt.tight_layout()
    plt.savefig("data/dataset_samples.png", dpi=100, bbox_inches='tight')
    plt.show()
    print("✅ Sample grid saved to data/dataset_samples.png")


if __name__ == "__main__":
    explore_dataset(DATA_DIR)