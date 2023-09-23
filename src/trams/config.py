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

LABELS_NAMES_MAP = {
    0: "accelerating_1_New",
    1: "accelerating_2_CKD_Long",
    2: "accelerating_3_CKD_Short",
    3: "accelerating_4_Old",
    4: "braking_1_New",
    5: "braking_2_CKD_Long",
    6: "braking_3_CKD_Short",
    7: "braking_4_Old",
}
