import os
import jsonlines
from pathlib import Path

import torchaudio
import pandas as pd
from torchaudio.backend.common import AudioMetaData
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from IPython.display import display

from trams.config import RAW_DATA_DIR_TRAIN, ARROW_DATA_DIR


def load_dataset_from_wav_files(test_split_pct: float, cached: bool = True):
    if ARROW_DATA_DIR.exists() and any(ARROW_DATA_DIR.iterdir()) and cached:
        return load_from_disk(ARROW_DATA_DIR)

    with jsonlines.open(RAW_DATA_DIR_TRAIN / "metadata.jsonl", mode="w") as writer:
        for idx, (root, _, files) in enumerate(os.walk(RAW_DATA_DIR_TRAIN)):
            if idx == 0:
                continue
            for file in files:
                path = Path(root) / file
                audio_metadata: AudioMetaData = torchaudio.info(path)
                relative_path = str(Path(Path(root).name) / file)
                metadata = {
                    "file_name": relative_path,
                    "sample_rate": audio_metadata.sample_rate,
                    "num_frames": audio_metadata.num_frames,
                    "num_channels": audio_metadata.num_channels,
                    "bits_per_sample": audio_metadata.bits_per_sample,
                }
                writer.write(metadata)

    dataset = load_dataset("audiofolder", data_dir=RAW_DATA_DIR_TRAIN, drop_labels=False)
    dataset = dataset["train"].train_test_split(test_split_pct, stratify_by_column="label", seed=100)
    dataset = DatasetDict({"train": dataset["train"], "validation": dataset["test"]})
    dataset.save_to_disk(ARROW_DATA_DIR)
    return dataset


def process_dataset(dataset: Dataset):
    def get_label_names(batch):
        train_dataset: Dataset = dataset["train"]
        return {"label_name": [train_dataset.features["label"].int2str(label) for label in batch["label"]]}

    dataset = dataset.map(get_label_names, batched=True)
    return dataset


def print_labels_statistics(train_dataset: Dataset):
    train_dataset.set_format("pandas")
    df = pd.concat([train_dataset["label_name"], train_dataset["label"]], axis=1)
    display(df.groupby(["label_name", "label"]).agg(count=("label", "count")))


def print_metadata_statistics(train_dataset: Dataset):
    train_dataset.set_format("pandas")
    df = pd.concat(
        [train_dataset["sample_rate"], train_dataset["bits_per_sample"], train_dataset["num_channels"]],
        axis=1,
    )
    display(
        df.groupby(["sample_rate", "bits_per_sample", "num_channels"]).agg(count=("sample_rate", "count"))
    )
