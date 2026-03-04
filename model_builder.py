# model_builder.py — Constructs the VGG16 transfer-learning model

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Resizing
from keras.applications import VGG16
from keras.optimizers import Adam

from config import IMG_SIZE, NUM_CLASSES


def build_model(img_size: int = IMG_SIZE, learning_rate: float = 1e-4):
    """
    Builds a VGG16-based transfer learning model for CIFAR-10.

    Architecture
    ------------
    Resizing → VGG16 (ImageNet weights, last 4 layers unfrozen)
    → Flatten → Dense(256) → Dropout → Dense(128) → Dropout
    → Dense(64) → Dropout → Dense(10, softmax)

    Parameters
    ----------
    img_size : int
        Target spatial resolution fed into VGG16 (default 224).
    learning_rate : float
        Adam learning rate (default 1e-4).

    Returns
    -------
    model : keras.Sequential
        Compiled and ready-to-train model.
    """
    print("Loading pretrained VGG16 (ImageNet weights)...")
    vgg = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(img_size, img_size, 3),
    )

    # Freeze all layers except the last 4 convolutional blocks
    print("Freezing base layers (keeping last 4 trainable)...")
    for layer in vgg.layers[:-4]:
        layer.trainable = False
    for layer in vgg.layers[-4:]:
        layer.trainable = True

    model = Sequential([
        Resizing(img_size, img_size),   # upscale 32×32 → 224×224
        vgg,
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(64,  activation="relu"),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation="softmax"),
    ], name="cifar10_vgg16")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("Model built successfully.")
    model.summary()
    return model
