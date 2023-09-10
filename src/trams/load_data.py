import os
import jsonlines
from pathlib import Path

import torchaudio
from torchaudio.backend.common import AudioMetaData
from datasets import load_dataset, load_from_disk

from trams.config import RAW_DATA_DIR_TRAIN, ARROW_DATA_DIR


def load_dataset_from_wav_files():
    if ARROW_DATA_DIR.exists() and any(ARROW_DATA_DIR.iterdir()):
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
    dataset.save_to_disk(ARROW_DATA_DIR)
    return dataset


dataset = load_dataset_from_wav_files()
# print(dataset["train"].features["label"].int2str)
