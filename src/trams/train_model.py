from lightning.pytorch.cli import LightningCLI

from trams.datamodule import TramsDataModule
from trams.model import TramsAudioClassifier


def main():
    cli = LightningCLI(TramsAudioClassifier, TramsDataModule)


if __name__ == "__main__":
    main()
