# Trams classification project

## Installation
This project requires Linux packages to be installed via the following command:
```bash
sudo apt-get install -y ffmpe sox llvm
```

## Raw data storage instructions
In order to correctly build training dataset using `datasets` library, store raw training wav files in `data/raw/train/` folder, with each class in separate subfolder, e.g. `Accelerating_2_CKD_Long`.
