# model_train.py — Trains the model and saves it to disk

import os
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from config import MODEL_PATH, EPOCHS, BATCH_SIZE
from data_loader import load_data
from model_builder import build_model


def train():
    """
    Full training pipeline:
      1. Load & preprocess CIFAR-10
      2. Build the VGG16 transfer-learning model
      3. Fit the model
      4. Save weights to MODEL_PATH
    """
    # ── Data ──────────────────────────────────────────────────────────────────
    X_train, y_train, X_test, y_test = load_data()

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model()

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\nStarting training — {EPOCHS} epochs, batch size {BATCH_SIZE}\n")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=1,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")

    # Final metrics
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy : {acc*100:.2f}%")
    print(f"Test loss     : {loss:.4f}")

    return history


if __name__ == "__main__":
    train()
