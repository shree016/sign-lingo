# backend/detector.py
#
# Real-time ASL detection using webcam + MediaPipe + trained model.
# Controls:
#   Q / ESC  — quit
#   SPACE    — add a space to the sentence
#   C        — clear the sentence
#   ENTER    — print the final sentence to terminal
#
# Run with: python backend/detector.py

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
from collections import deque, Counter
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── CONFIG ─────────────────────────────────────────────────────────────────────
MODEL_PATH       = "model/sign_model.keras"
CLASSES_PATH     = "model/classes.npy"
MODEL_PATH_TASK  = "data/hand_landmarker.task"

CONFIDENCE_THRESHOLD = 0.80
SMOOTHING_FRAMES     = 10
HOLD_FRAMES          = 20

# BGR colours for OpenCV drawing
GREEN  = (0, 220, 100)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
YELLOW = (0, 215, 255)
GRAY   = (160, 160, 160)
# ──────────────────────────────────────────────────────────────────────────────


def normalize_landmarks(hand_lms):
    """
    Identical normalization to extract_landmarks.py.
    MUST stay the same — model was trained on this format.
    """
    coords   = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms])
    wrist    = coords[0]
    coords   = coords - wrist
    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 0:
        coords = coords / max_dist
    return coords.flatten().tolist()


def draw_landmarks_on_frame(frame, hand_lms):
    """
    Draws 21 landmark dots and skeleton connections on the frame.
    hand_lms: list of NormalizedLandmark objects from MediaPipe.
    """
    h, w = frame.shape[:2]

    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17)
    ]

    # Convert 0-1 normalized coords → pixel coords
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]

    # Draw bone connections
    for a, b in connections:
        cv2.line(frame, pts[a], pts[b], (100, 200, 100), 2, cv2.LINE_AA)

    # Draw joint dots
    for i, (px, py) in enumerate(pts):
        if i in [4, 8, 12, 16, 20]:   # Fingertips — bigger green dot
            cv2.circle(frame, (px, py), 7, GREEN, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 7, WHITE,  1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (px, py), 4, WHITE, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 4, GRAY,   1, cv2.LINE_AA)


def draw_ui(frame, prediction, confidence, sentence, hold_progress, no_hand):
    """
    Draws all HUD overlays:
      - Semi-transparent top bar with predicted letter + confidence
      - Green hold-progress bar
      - Semi-transparent bottom bar with sentence + controls
    """
    h, w = frame.shape[:2]

    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), BLACK, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Bottom bar
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 80), (w, h), BLACK, -1)
    cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)

    if no_hand:
        cv2.putText(frame, "No hand detected", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, GRAY, 2, cv2.LINE_AA)
    else:
        if prediction and confidence >= CONFIDENCE_THRESHOLD:
            cv2.putText(frame, str(prediction), (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.2, GREEN, 3, cv2.LINE_AA)
            cv2.putText(frame, f"{confidence*100:.1f}%", (120, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Hold still...", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, YELLOW, 2, cv2.LINE_AA)

    # Hold progress bar
    bar_w = int((hold_progress / HOLD_FRAMES) * (w - 40))
    cv2.rectangle(frame, (20, 75), (w - 20, 82), GRAY, -1)
    if bar_w > 0:
        cv2.rectangle(frame, (20, 75), (20 + bar_w, 82), GREEN, -1)

    # Sentence
    cv2.putText(frame, f"Sentence: {sentence}_", (20, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)

    # Controls
    cv2.putText(frame, "Q=quit  SPACE=space  C=clear  ENTER=print",
                (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRAY, 1,
                cv2.LINE_AA)


class SignDetector:
    """
    Loads the Keras model and MediaPipe hand landmarker.
    predict_frame() runs the full pipeline on one BGR frame.
    """

    def __init__(self):
        print("Loading model...")
        self.model   = keras.models.load_model(MODEL_PATH)
        self.classes = np.load(CLASSES_PATH, allow_pickle=True)
        # Warm up the model — first call is always slow due to TF graph building
        dummy = np.zeros((1, 63), dtype=np.float32)
        self.model(dummy, training=False)
        print(f"  ✅ Model loaded — {len(self.classes)} classes")

        print("Loading MediaPipe hand landmarker...")
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python.vision import (
            HandLandmarker, HandLandmarkerOptions, RunningMode
        )

        options = HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=MODEL_PATH_TASK
            ),
            running_mode=RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = HandLandmarker.create_from_options(options)
        print("  ✅ MediaPipe loaded")

        self.pred_buffer = deque(maxlen=SMOOTHING_FRAMES)

    def predict_frame(self, frame_bgr):
        """
        Full pipeline on one BGR frame.

        Returns:
            hand_lms   : list of NormalizedLandmark (or None)
            prediction : raw predicted class string (or None)
            confidence : float 0-1
            smoothed   : smoothed prediction from buffer (or None)
        """
        # BGR → RGB → MediaPipe Image
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = self.detector.detect(mp_image)

        if not result.hand_landmarks:
            self.pred_buffer.clear()
            return None, None, 0.0, None

        # Use first detected hand
        hand_lms = result.hand_landmarks[0]

        # Normalize → model input
        normalized  = normalize_landmarks(hand_lms)
        input_tensor = tf.constant([normalized], dtype=tf.float32)  # (1, 63)

        # Direct model call — faster than model.predict() in a loop
        logits = self.model(input_tensor, training=False)
        probs  = tf.nn.softmax(logits[0]).numpy()

        pred_idx   = int(np.argmax(probs))
        prediction = str(self.classes[pred_idx])
        confidence = float(probs[pred_idx])

        # Add confident predictions to smoothing buffer
        if confidence >= CONFIDENCE_THRESHOLD:
            self.pred_buffer.append(prediction)

        # Smoothed = most common prediction across last N frames
        smoothed = None
        if len(self.pred_buffer) >= 3:
            smoothed = Counter(self.pred_buffer).most_common(1)[0][0]

        return hand_lms, prediction, confidence, smoothed

    def close(self):
        self.detector.close()


def run():
    """Main webcam loop."""

    print("\n" + "=" * 50)
    print("  Sign Lingo — Real-Time Detection")
    print("=" * 50)

    sign_detector = SignDetector()

    # Open default webcam — change 0 to 1 or 2 if wrong camera opens
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        print("   Try changing VideoCapture(0) to VideoCapture(1)")
        sign_detector.close()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n✅ Webcam opened — show your hand to the camera!")
    print("   Controls: Q=quit  SPACE=space  C=clear  ENTER=print\n")

    sentence         = ""
    last_added       = None
    hold_counter     = 0
    current_smoothed = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        # Flip horizontally — mirror view feels natural
        frame = cv2.flip(frame, 1)

        # Run full detection pipeline
        hand_lms, prediction, confidence, smoothed = \
            sign_detector.predict_frame(frame)

        # Draw hand skeleton if detected
        if hand_lms is not None:
            draw_landmarks_on_frame(frame, hand_lms)

        # Hold-to-confirm logic
        if smoothed and smoothed == current_smoothed:
            hold_counter = min(hold_counter + 1, HOLD_FRAMES)
        else:
            hold_counter     = 0
            current_smoothed = smoothed

        # Commit letter to sentence after holding long enough
        if hold_counter >= HOLD_FRAMES and smoothed and smoothed != last_added:
            if smoothed == "space":
                sentence += " "
            else:
                sentence += smoothed
            last_added   = smoothed
            hold_counter = 0
            print(f"  Added: '{smoothed}' → sentence: '{sentence}'")

        # Draw HUD
        draw_ui(
            frame,
            prediction    = smoothed,
            confidence    = confidence,
            sentence      = sentence,
            hold_progress = hold_counter,
            no_hand       = (hand_lms is None)
        )

        cv2.imshow("Sign Lingo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):       # Q or ESC
            print("\nQuitting...")
            break
        elif key == ord('c'):
            sentence   = ""
            last_added = None
            print("  Sentence cleared")
        elif key == 13:                 # ENTER
            print(f"\n  Final sentence: '{sentence}'\n")
        elif key == ord(' '):
            sentence  += " "
            last_added = None

    cap.release()
    cv2.destroyAllWindows()
    sign_detector.close()
    print(f"\nFinal sentence: '{sentence}'")
    print("Goodbye!")


if __name__ == "__main__":
    run()