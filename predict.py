# predict.py — Load the trained model and predict on random CIFAR-10 test images

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import random
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model
from keras.layers import Lambda
from keras.datasets import cifar10
from keras.applications.vgg16 import preprocess_input

from config import MODEL_PATH, CLASS_NAMES, GDRIVE_MODEL_URL


def _download_model():
    """
    Downloads the pre-trained model from Google Drive if it is not present locally.
    Requires:  pip install gdown
    """
    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required to auto-download the model.\n"
            "Install it with:  pip install gdown"
        )

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"Downloading pre-trained model from Google Drive...")
    gdown.download(GDRIVE_MODEL_URL, MODEL_PATH, quiet=False)
    print(f"Model saved to {MODEL_PATH}")


def predict_random(n: int = 1, show_plot: bool = True):
    """
    Picks `n` random images from the CIFAR-10 test set, runs inference,
    and prints / plots the results.

    Parameters
    ----------
    n : int
        Number of random images to predict (default 1).
    show_plot : bool
        Whether to display the image with matplotlib (default True).
    """
    # ── Model ─────────────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at '{MODEL_PATH}'.")
        _download_model()

    print("Loading model...")
    model = load_model(
        MODEL_PATH,
        custom_objects={"preprocess_input": preprocess_input, "Lambda": Lambda},
    )
    print("Model loaded.\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    (_, _), (X_test, y_test) = cifar10.load_data()

    for i in range(n):
        idx = random.randint(0, len(X_test) - 1)
        img_raw = X_test[idx]                          # original uint8 image
        actual  = CLASS_NAMES[int(y_test[idx][0])]

        # Preprocess for model
        img_input = img_raw.astype("float32")
        img_input = np.expand_dims(img_input, axis=0)  # (1, 32, 32, 3)
        img_input = preprocess_input(img_input)

        # Inference
        pred       = model.predict(img_input, verbose=0)
        pred_class = np.argmax(pred)
        confidence = pred[0][pred_class] * 100

        print(f"[{i+1}/{n}]  Actual: {actual:<12}  "
              f"Predicted: {CLASS_NAMES[pred_class]:<12}  "
              f"Confidence: {confidence:.1f}%")

        if show_plot:
            plt.figure(figsize=(3, 3))
            plt.imshow(img_raw)
            plt.title(
                f"Actual: {actual}\n"
                f"Pred: {CLASS_NAMES[pred_class]} ({confidence:.1f}%)",
                fontsize=9,
            )
            plt.axis("off")
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    predict_random(n=1)
