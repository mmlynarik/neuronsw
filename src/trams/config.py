from pathlib import Path

ROOT_DIR = Path.cwd()
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_DIR_TRAIN = RAW_DATA_DIR / "train"
RAW_DATA_DIR_TEST = RAW_DATA_DIR / "test"
ARROW_DATA_DIR = DATA_DIR / "arrow"
