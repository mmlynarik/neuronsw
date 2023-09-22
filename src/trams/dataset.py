import os
import jsonlines
from pathlib import Path
from typing import Any, Callable
import random

import torchaudio
import pandas as pd
import torch as pt
from torchaudio.backend.common import AudioMetaData
from torchaudio import transforms
from datasets import load_dataset, Dataset, DatasetDict
from IPython.display import display

from trams.config import RAW_DATA_DIR_TRAIN, ARROW_DATA_DIR, NUM_FFT, NUM_MELS, MAX_DB


Batch = dict[str, Any]


def _add_label_name(batch: Batch, int2str: Callable) -> Batch:
    return {"label_name": [int2str(label) for label in batch["label"]]}


def _get_max_frames(dataset: Dataset, max_length_secs: int) -> int:
    sample_rate = dataset["train"]["sample_rate"][0]
    return int(max_length_secs * sample_rate)


def _flatten_example_dict(batch: Batch) -> Batch:
    return {
        "audio": [items["array"] for items in batch["audio"]],
        "path": [items["path"] for items in batch["audio"]],
    }


def _sample_audio_length_for_negatives(batch: Batch, sample_rate: int) -> Batch:
    batch_size = len(batch["audio"])
    sampled_lengths = pt.normal(mean=4, std=1, size=(batch_size,))  # 4, 1 = mean, std of audio lengths dist
    return {
        "audio": [
            audio[: int(sampled_length.item() * sample_rate)] if label == 8 else audio
            for audio, label, sampled_length in zip(batch["audio"], batch["label"], sampled_lengths)
        ],
        "num_frames": [int(sampled_length.item() * sample_rate) for sampled_length in sampled_lengths],
    }


def _add_gaussian_white_noise_to_one_audio(audio: list[float], snr: float) -> list[float]:
    """When mean = 0, std_noise = rms_noise because definition of rms signifies definition of std."""
    rms_signal = pt.sqrt(pt.mean(pt.tensor(audio) ** 2))
    rms_noise = pt.sqrt(rms_signal**2 / (pow(10, snr / 10)))  # from definition of signal-to-noise ratio
    std_noise = rms_noise
    return (pt.tensor(audio) + pt.normal(0, std_noise, (len(audio),))).tolist()


def _add_gaussian_noise(batch: Batch, snr: float) -> Batch:
    return {"audio": [_add_gaussian_white_noise_to_one_audio(audio, snr) for audio in batch["audio"]]}


def _truncate(batch: Batch, max_frames: int) -> Batch:
    return {"audio": [audio[:max_frames] for audio in batch["audio"]]}


def _pad_one_audio(audio: list[float], max_frames: int) -> list[float]:
    padding_size = max_frames - len(audio)
    padding_left = random.randint(0, padding_size)
    padding_right = padding_size - padding_left
    return [0] * padding_left + audio + [0] * padding_right


def _pad(batch: Batch, max_frames: int) -> Batch:
    return {"audio": [_pad_one_audio(audio, max_frames) for audio in batch["audio"]]}


def _get_mel_spectrogram(
    batch: Batch, mel_spectrogram: transforms.MelSpectrogram, amplitude_transformer: transforms.AmplitudeToDB
) -> Batch:
    return {"spectrogram": [amplitude_transformer(mel_spectrogram(audio)) for audio in batch["audio"]]}


def load_dataset_from_wav_files() -> DatasetDict:
    """Generates metadata.jsonl file and then loads wav files into DatasetDict using that metadata."""
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
    return dataset


def split_dataset(dataset: DatasetDict, validation_pct: float) -> DatasetDict:
    dataset = dataset["train"].train_test_split(validation_pct, stratify_by_column="label", seed=100)
    return DatasetDict({"train": dataset["train"], "validation": dataset["test"]})


def process_dataset(dataset: DatasetDict, max_length_secs: int, snr: float) -> DatasetDict:
    train_dataset: Dataset = dataset["train"]
    int2str = train_dataset.features["label"].int2str
    max_frames = _get_max_frames(dataset, max_length_secs)
    sample_rate = train_dataset["sample_rate"][0]
    mel_spectrogram = transforms.MelSpectrogram(sample_rate, n_fft=NUM_FFT, n_mels=NUM_MELS)
    amplitude_transformer = transforms.AmplitudeToDB(top_db=MAX_DB)

    dataset = (
        dataset.map(_add_label_name, batched=True, fn_kwargs={"int2str": int2str}, desc="Add label name")
        .map(_flatten_example_dict, batched=True, remove_columns=["audio"], desc="Flatten example dict")
        .map(
            _sample_audio_length_for_negatives,
            batched=True,
            fn_kwargs={"sample_rate": sample_rate},
            desc="Sample audio lengths for negatives",
        )
        .map(_add_gaussian_noise, batched=True, fn_kwargs={"snr": snr}, desc="Add gaussian white noise")
        .map(_truncate, batched=True, fn_kwargs={"max_frames": max_frames}, desc="Truncate")
        .map(_pad, batched=True, fn_kwargs={"max_frames": max_frames}, desc="Pad")
        .with_format("torch", columns=["audio", "label"])
        .map(
            _get_mel_spectrogram,
            batched=True,
            fn_kwargs={"mel_spectrogram": mel_spectrogram, "amplitude_transformer": amplitude_transformer},
            desc="Generate Mel Spectrogram",
        )
    )
    dataset.save_to_disk(ARROW_DATA_DIR)
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
