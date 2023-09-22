# Trams audio classification project

## 1. Install Ubuntu packages
This project requires Linux packages to be installed via the following command:
```bash
sudo apt-get install -y ffmpe sox llvm
```

## 2. Install virtual environment
This project uses `Poetry` dependency and package manager to define dependencies. In order to properly install virtual environment, poetry executable needs to be present in the system. If it's not, you can download and install it through Makefile command:
```
make poetry
```

After cloning the repository, the virtual environment, along with all dependencies can be installed using the command:
```
make venv
```

## 3. Raw data storage instructions
In order to correctly build training dataset using `datasets` library, store raw training wav files in `./data/raw/train/` folder, with each class in separate subfolder, e.g. `./data/raw/train/Accelerating_2_CKD_Long`. Test files should be placed into `data/raw/test` folder.
