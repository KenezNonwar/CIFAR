# CIFAR-10 Image Classifier — VGG16 Transfer Learning

A deep learning image classifier trained on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset using VGG16 transfer learning. Achieves **~88% test accuracy**.

---

## What It Does

Given a random image from the CIFAR-10 test set, the model predicts which of 10 classes it belongs to:

`Airplane` · `Automobile` · `Bird` · `Cat` · `Deer` · `Dog` · `Frog` · `Horse` · `Ship` · `Truck`

---

## Project Structure

```
CIFAR/
├── config.py          # Central config (image size, paths, class names, Drive link)
├── data_loader.py     # Loads and preprocesses CIFAR-10
├── model_builder.py   # Builds the VGG16 transfer learning model
├── model_train.py     # Training pipeline (skip if using pre-trained model)
├── predict.py         # Run inference on a random test image
├── requirements.txt   # Dependencies
└── models/            # Auto-created on first run (ignored by Git)
```

---

## Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/KenezNonwar/CIFAR.git
cd CIFAR
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run prediction
```bash
python predict.py
```

On the **first run**, the pre-trained model (~180MB) will be automatically downloaded from Google Drive and saved to `models/`. Every run after that skips the download.

---

## Model

- **Architecture:** VGG16 (ImageNet weights) + custom classification head
- **Trainable layers:** Last 4 layers of VGG16 unfrozen
- **Head:** `Flatten → Dense(256) → Dropout → Dense(128) → Dropout → Dense(64) → Dropout → Softmax(10)`
- **Optimizer:** Adam (lr=0.0001)
- **Input size:** 32×32 upscaled to 224×224 via `Resizing` layer
- **Test accuracy:** ~88%

> The pre-trained model is hosted on Google Drive and auto-downloaded on first run via `gdown`.

---

## Train From Scratch

If you want to retrain the model yourself (note: slow without GPU):

```bash
python model_train.py
```

This will train for 5 epochs with batch size 4 and save the model to `models/cifar_vgg_model.keras`.

---

## Requirements

- Python 3.8+
- TensorFlow 2.12+
- Keras
- NumPy
- Matplotlib
- gdown

Install all with:
```bash
pip install -r requirements.txt
```

---

## Troubleshooting

**gdown quota error on model download?**
```bash
gdown --id 1NoBibkBfMD8e8aagRuv8c72V9ixC1pxG -O models/cifar_vgg_model.keras
```
Then run `python predict.py` as normal.

---

## Author

**KenezNonwar** — [github.com/KenezNonwar](https://github.com/KenezNonwar)
