# backend/extract_landmarks.py  (updated for MediaPipe 0.10.x+ Tasks API)
#
# The new API uses mediapipe.tasks instead of mediapipe.solutions.
# Key differences:
#   OLD: mp.solutions.hands.Hands()
#   NEW: HandLandmarker with a downloaded .task model file
#
# Run with: python backend/extract_landmarks.py

import os
import csv
import urllib.request
import cv2
import numpy as np

# New Tasks-based imports
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

try:
    from tqdm import tqdm
except ImportError:
    os.system("pip install tqdm")
    from tqdm import tqdm


# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR   = "data/raw/asl_alphabet_train/asl_alphabet_train"
OUTPUT_CSV = "data/landmarks/landmarks.csv"
MODEL_PATH = "data/hand_landmarker.task"   # Downloaded once, reused

CLASSES_TO_USE = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["space"]
# ──────────────────────────────────────────────────────────────────────────────


def download_model():
    """
    The new MediaPipe Tasks API requires a .task model file.
    We download it once from Google's servers and cache it locally.
    File size: ~5MB
    """
    if os.path.exists(MODEL_PATH):
        print(f"✅ Model already downloaded at {MODEL_PATH}")
        return

    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
    print(f"Downloading hand landmarker model (~5MB)...")
    os.makedirs("data", exist_ok=True)
    urllib.request.urlretrieve(url, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")


def build_csv_header():
    """63 coordinate columns + label column."""
    header = []
    for i in range(21):
        header += [f"x{i}", f"y{i}", f"z{i}"]
    header.append("label")
    return header


def normalize_landmarks(landmarks):
    """
    Normalizes 21 landmarks relative to the wrist (landmark 0).
    Works identically to before — the landmark objects just come
    from a different source now.

    landmarks: list of NormalizedLandmark objects, each with .x .y .z
    returns:   flat list of 63 floats
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

    # Shift origin to wrist
    wrist  = coords[0]
    coords = coords - wrist

    # Scale by max distance from wrist
    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 0:
        coords = coords / max_dist

    return coords.flatten().tolist()


def extract_landmarks_from_dataset():
    """
    Loops through every image in the dataset, runs the MediaPipe
    Tasks HandLandmarker on each one, and saves results to CSV.
    """

    # ── Download model if needed ───────────────────────────────────────────────
    download_model()

    # ── Create the HandLandmarker ──────────────────────────────────────────────
    # The new API uses a builder pattern with options
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)

    options = HandLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.IMAGE,   # Static images (not live video)
        num_hands=1,                       # Only detect one hand
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # ── Setup output ───────────────────────────────────────────────────────────
    os.makedirs("data/landmarks", exist_ok=True)

    success_count = 0
    skip_count    = 0
    error_count   = 0

    # HandLandmarker is used as a context manager (with statement)
    # This ensures resources are properly cleaned up when done
    with HandLandmarker.create_from_options(options) as detector:
        with open(OUTPUT_CSV, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(build_csv_header())

            for class_name in CLASSES_TO_USE:
                class_path = os.path.join(DATA_DIR, class_name)

                if not os.path.isdir(class_path):
                    print(f"⚠️  Skipping '{class_name}' — folder not found")
                    continue

                image_files = [
                    f for f in os.listdir(class_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ]

                print(f"\nProcessing '{class_name}' ({len(image_files)} images)...")

                for img_file in tqdm(image_files, desc=class_name, unit="img"):
                    img_path = os.path.join(class_path, img_file)

                    # ── Load image ─────────────────────────────────────────────
                    img_bgr = cv2.imread(img_path)
                    if img_bgr is None:
                        error_count += 1
                        continue

                    # Convert BGR → RGB (MediaPipe always needs RGB)
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                    # ── Wrap in MediaPipe Image object ─────────────────────────
                    # The new API wraps numpy arrays in an mp.Image object
                    # ImageFormat.SRGB tells it the color space
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=img_rgb
                    )

                    # ── Run detection ──────────────────────────────────────────
                    # detect() returns a HandLandmarkerResult object
                    result = detector.detect(mp_image)

                    # result.hand_landmarks is a list of detected hands
                    # Each hand is a list of 21 NormalizedLandmark objects
                    if not result.hand_landmarks:
                        skip_count += 1
                        continue

                    # Take the first detected hand
                    hand_landmarks = result.hand_landmarks[0]

                    # ── Normalize and save ─────────────────────────────────────
                    normalized = normalize_landmarks(hand_landmarks)
                    row = normalized + [class_name]
                    writer.writerow(row)
                    success_count += 1

    # ── Final report ───────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"✅ Extraction complete!")
    print(f"   Saved to:     {OUTPUT_CSV}")
    print(f"   Extracted:    {success_count:,} landmarks")
    print(f"   Skipped:      {skip_count:,} (no hand detected)")
    print(f"   Errors:       {error_count:,} (unreadable files)")
    total = success_count + skip_count + error_count
    rate  = (success_count / total * 100) if total > 0 else 0
    print(f"   Success rate: {rate:.1f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    extract_landmarks_from_dataset()