# Prague Trams Audio Classification Project

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


## 4. Train model
To train model, the `WANDB_API_KEY` environment variable must be set to W&B account token used for logging the experiment. [Weights & Biases (W&B)](https://www.wandb.ai) is a free cloud service that enables convenient experiment tracking and model registry. Use `.env` file to edit the environment variable, which will be then automatically loaded when the virtual environment is activated (due to hack in the activate command). Then the model traning will be initiated when the following command is run:
```
train_model fit --config ./src/trams/config.yaml
```

## 5. Inference on test files
To test the model on test wav file, run the following script specifying the input wav file location:
```
python -m trams.predict --input-wav ./data/raw/test/tram-2018-12-07-15-32-08.wav
```
