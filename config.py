# config.py — Central configuration for the CIFAR-10 VGG16 project

# ── Image & Model ─────────────────────────────────────────────────────────────
IMG_SIZE    = 224
NUM_CLASSES = 10
BATCH_SIZE  = 4
EPOCHS      = 5

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH        = "models/cifar_vgg_model.keras"

# Google Drive direct-download link for the pre-trained model (>180 MB).
# Replace the value below with your actual shareable link.
# How to get it:
#   1. Right-click the file in Drive → "Get link" → set to "Anyone with the link"
#   2. Copy the file ID from the URL:
#      https://drive.google.com/file/d/<FILE_ID>/view
#   3. Paste the FILE_ID below.
GDRIVE_FILE_ID    = "1NoBibkBfMD8e8aagRuv8c72V9ixC1pxG"
GDRIVE_MODEL_URL  = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

# ── Classes ───────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog","Frog","Horse","Ship","Truck",
]
