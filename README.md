# Trams classification project

## 1. Install Linux packages
This project requires Linux packages to be installed via the following command:
```bash
sudo apt-get install -y ffmpe sox llvm
```

## 2. Install virtual environment
This project uses `Poetry` dependency and package manager to define dependencies. In order to install virtual environment, poetry needs to be installed. If it's not, you can install it through Makefile command:
```
make poetry
```

Then the virtual environment can be installed using the command:
```
make venv
```

## Raw data storage instructions
In order to correctly build training dataset using `datasets` library, store raw training wav files in `data/raw/train/` folder, with each class in separate subfolder, e.g. `Accelerating_2_CKD_Long`.
