from pathlib import Path

ROOT_DIR = Path.cwd()
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_DIR_TRAIN = RAW_DATA_DIR / "train"
RAW_DATA_DIR_TEST = RAW_DATA_DIR / "test"
ARROW_DATA_DIR = DATA_DIR / "arrow"

MAX_LENGTH_SECS = 4
ONE_TENTH_SEC = 0.1
NUM_FFT = 1024
NUM_MELS = 64
MAX_DB = 90

TRAINED_MODEL_PATH = ROOT_DIR / "src" / "trams" / "trained_model" / "model.ckpt"

TRAMS_LABELS_NAMES = [
    "accelerating_1_New",
    "accelerating_2_CKD_Long",
    "accelerating_3_CKD_Short",
    "accelerating_4_Old",
    "braking_1_New",
    "braking_2_CKD_Long",
    "braking_3_CKD_Short",
    "braking_4_Old",
]
