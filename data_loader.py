# data_loader.py — Loads and preprocesses the CIFAR-10 dataset

import numpy as np
from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from keras.applications.vgg16 import preprocess_input

from config import NUM_CLASSES


def load_data(verbose: bool = True):
    """
    Downloads (if needed) and returns pre-processed CIFAR-10 splits.

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray
        X arrays are VGG16-preprocessed float32 images (0-255 range kept,
        channel means subtracted by preprocess_input).
        y arrays are one-hot encoded with shape (N, NUM_CLASSES).
    """
    if verbose:
        print("Loading CIFAR-10 dataset...")

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Cast to float32 before preprocessing
    X_train = X_train.astype("float32")
    X_test  = X_test.astype("float32")

    # VGG16 preprocessing (subtracts ImageNet channel means)
    X_train = preprocess_input(X_train)
    X_test  = preprocess_input(X_test)

    # One-hot encode labels
    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test  = to_categorical(y_test,  NUM_CLASSES)

    if verbose:
        print(f"  Train images : {X_train.shape}  Labels: {y_train.shape}")
        print(f"  Test  images : {X_test.shape}   Labels: {y_test.shape}")

    return X_train, y_train, X_test, y_test
