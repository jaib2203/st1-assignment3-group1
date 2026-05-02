# config.py
# This is where configuration data is stored for the project.

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = Path(r"data\raw")
DATA_OUTPUT_DIR = Path(r"data\processed")
EDA_OUTPUT_DIR = Path(r"outputs\eda")
IMAGE_SIZE = (128, 128)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}