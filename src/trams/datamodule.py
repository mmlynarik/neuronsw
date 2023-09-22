import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from datasets import load_from_disk, DatasetDict
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

from trams.dataset import load_dataset_from_wav_files, process_dataset, split_dataset
from trams.config import RAW_DATA_DIR_TRAIN, ARROW_DATA_DIR

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class TramsDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        validation_split: float,
        max_length_secs: int,
        snr: float,
        raw_data_dir: Path = RAW_DATA_DIR_TRAIN,
        arrow_data_dir: Path = ARROW_DATA_DIR,
        use_cache: bool = True,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.max_length_secs = max_length_secs
        self.snr = snr
        self.raw_data_dir = raw_data_dir
        self.arrow_data_dir = arrow_data_dir
        self.use_cache = use_cache

    def prepare_data(self) -> DatasetDict:
        if Path(self.arrow_data_dir / "dataset_dict.json").exists():
            if self.use_cache:
                log.info(f"Processed dataset already exists in {self.arrow_data_dir} folder.")
                return load_from_disk(self.arrow_data_dir)
            else:
                shutil.rmtree(ARROW_DATA_DIR)
                os.makedirs(ARROW_DATA_DIR)

        if not self.raw_data_dir.exists():
            log.error(f"Raw wav files were not found. Please supply them in {self.raw_data_dir}")
            return

        dataset = load_dataset_from_wav_files()
        dataset = split_dataset(dataset, validation_pct=self.validation_split)
        dataset = process_dataset(dataset, self.max_length_secs, self.snr)
        return dataset

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = load_from_disk(self.arrow_data_dir)
        dataset = dataset.remove_columns(
            ["sample_rate", "num_frames", "num_channels", "bits_per_sample", "path", "label_name", "audio"]
        )

        if stage is None or stage == "fit":
            self.train_dataset = dataset["train"]
        if stage is None or stage in ["fit", "validate"]:
            self.val_dataset = dataset["validation"]
        if stage is None or stage == "test":
            pass

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass


def datamodule_sanity_check():
    dm = TramsDataModule(batch_size=16, validation_split=0.1, max_length_secs=4, snr=5)
    dm.prepare_data()
    dm.setup()

    for step, batch in enumerate(dm.val_dataloader()):
        if step == 0:
            print(batch)
            break


if __name__ == "__main__":
    datamodule_sanity_check()
