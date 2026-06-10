# backend/model.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(num_classes: int, input_dim: int = 63) -> keras.Model:
    """
    Builds and returns the Sign Lingo classification model.

    Architecture:
    - Input:      63 normalized landmark coordinates
    - Dense 512:  Learns broad hand shape patterns
    - Dense 256:  Learns finer distinctions between signs
    - Dense 128:  Distills the most important features
    - Output:     num_classes neurons — one probability per sign

    Args:
        num_classes: How many sign classes to predict (27 in our case)
        input_dim:   Number of input features (63 landmark coordinates)

    Returns:
        A compiled Keras model ready for training
    """

    model = keras.Sequential([

        # Input layer — expects a flat array of 63 floats
        layers.Input(shape=(input_dim,)),

        # First hidden layer — learns broad hand shape patterns
        # 512 neurons × 63 inputs = 32,256 learnable weights
        layers.Dense(512),
        layers.BatchNormalization(momentum=0.9),
        layers.Activation('relu'),
        layers.Dropout(0.4),   # Drop 40% of neurons during training

        # Second hidden layer — refines patterns learned above
        layers.Dense(256),
        layers.BatchNormalization(momentum=0.9),
        layers.Activation('relu'),
        layers.Dropout(0.3),   # Drop 30%

        # Third hidden layer — final feature extraction
        layers.Dense(128),
        layers.BatchNormalization(momentum=0.9),
        layers.Activation('relu'),
        layers.Dropout(0.2),   # Drop 20%

        # Output layer — one neuron per class, no activation here
        # SparseCategoricalCrossentropy(from_logits=True) handles it
        layers.Dense(num_classes)

    ], name="sign_lingo_model")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":
    model = build_model(num_classes=27)
    model.summary()