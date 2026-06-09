# backend/utils.py
# Verifies all dependencies are installed correctly.
# Run once after setup: python backend/utils.py

def check_dependencies():
    print("Checking all dependencies...\n")

    import sys
    print(f"Python:       {sys.version.split()[0]}")

    import flask
    print(f"Flask:        {flask.__version__}")

    import cv2
    print(f"OpenCV:       {cv2.__version__}")

    import mediapipe as mp
    print(f"MediaPipe:    {mp.__version__}")

    import tensorflow as tf
    print(f"TensorFlow:   {tf.__version__}")

    import numpy as np
    print(f"NumPy:        {np.__version__}")

    import pandas as pd
    print(f"Pandas:       {pd.__version__}")

    import sklearn
    print(f"Scikit-learn: {sklearn.__version__}")

    import matplotlib
    print(f"Matplotlib:   {matplotlib.__version__}")

    import PIL
    print(f"Pillow:       {PIL.__version__}")

    print("\n✅ All dependencies loaded successfully!")


if __name__ == "__main__":
    check_dependencies()