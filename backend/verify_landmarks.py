# backend/verify_landmarks.py
#
# Quick sanity check on the extracted landmarks CSV.
# Run with: python backend/verify_landmarks.py

import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "data/landmarks/landmarks.csv"

def verify():
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)

    print(f"\n📐 Shape: {df.shape}")
    print(f"   Rows (samples): {df.shape[0]:,}")
    print(f"   Cols (features + label): {df.shape[1]}")

    print(f"\n🏷️  Label column — unique classes:")
    print(f"   {sorted(df['label'].unique())}")

    print(f"\n📊 Samples per class:")
    counts = df['label'].value_counts().sort_index()
    for label, count in counts.items():
        bar = "█" * (count // 200)
        print(f"  {label:6}  {count:5,}  {bar}")

    print(f"\n🔍 First row preview (first 9 values + label):")
    first_row = df.iloc[0]
    coords = [f"{v:.4f}" for v in first_row[:9]]
    print(f"   {coords} ... label='{first_row['label']}'")

    print(f"\n✅ Range check (should be roughly -1 to 1):")
    numeric_cols = df.columns[:-1]   # All except 'label'
    print(f"   Min value: {df[numeric_cols].min().min():.4f}")
    print(f"   Max value: {df[numeric_cols].max().max():.4f}")

    # Plot class distribution
    plt.figure(figsize=(14, 5))
    counts.plot(kind='bar', color='steelblue', edgecolor='white')
    plt.title("Samples per class in landmarks.csv")
    plt.xlabel("Sign")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("data/landmark_distribution.png", dpi=100)
    plt.show()
    print("\n📸 Distribution chart saved to data/landmark_distribution.png")

if __name__ == "__main__":
    verify()