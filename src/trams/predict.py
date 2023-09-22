from pathlib import Path

import torchaudio
import torch as pt
import pandas as pd
from torchaudio.backend.common import AudioMetaData
from torchaudio import transforms


from trams.config import (
    NUM_FFT,
    NUM_MELS,
    MAX_DB,
    MAX_LENGTH_SECS,
    ONE_TENTH_SEC,
    TRAINED_MODEL_PATH,
)
from trams.model import TramsAudioClassifier, ModelConfig
from trams.datamodule import TramsDataModule


def predict_trams_from_wav(input_wav: Path, output_csv: Path):
    audio, sample_rate = torchaudio.load(input_wav)
    audio_metadata: AudioMetaData = torchaudio.info(input_wav)
    if sample_rate != 22050 or audio_metadata.num_channels != 1:
        raise ValueError("Current model only supports sample rate of 22050 and mono channel.")

    num_frames = audio_metadata.num_frames
    num_seconds = num_frames / sample_rate
    print(f"Loaded {input_wav.parts[-1]} file spanning {num_seconds} seconds.")

    mel_spectrogram = transforms.MelSpectrogram(sample_rate, n_fft=NUM_FFT, n_mels=NUM_MELS)
    amplitude_transformer = transforms.AmplitudeToDB(top_db=MAX_DB)

    slice_frames = int(MAX_LENGTH_SECS * sample_rate)
    offset_frames = int(ONE_TENTH_SEC * sample_rate)

    checkpoint = pt.load(TRAINED_MODEL_PATH)
    model = TramsAudioClassifier(ModelConfig())
    model.load_state_dict(checkpoint["state_dict"])

    predictions = []
    model.eval()
    with pt.no_grad():
        position = 0
        while position <= num_frames - slice_frames:
            slice = audio[:, position : position + slice_frames]
            spectrogram = amplitude_transformer(mel_spectrogram(slice))
            output: pt.Tensor = model(spectrogram)
            prediction = pt.argmax(output.softmax(dim=1), dim=1).item()
            if prediction == 8:
                position += offset_frames
            else:
                predictions.append((position / sample_rate, prediction))
                position += slice_frames
    # [(24.7, 7), (28.7, 7), (43.8, 7), (47.8, 7), (65.7, 4), (69.7, 4), (102.9, 4), (106.9, 4), (139.1, 3), (143.1, 3), (175.4, 4), (179.4, 4)]
    output_mask = [True]
    idx = 0
    while idx + 1 < len(predictions):
        if (predictions[idx + 1][0] - predictions[idx][0] == MAX_LENGTH_SECS) and (
            predictions[idx + 1][1] == predictions[idx][1]
        ):
            output_mask.append(False)
        else:
            output_mask.append(True)
        idx += 1
    df = pd.DataFrame(predictions, columns=["seconds_offset", "label"])[output_mask].reset_index(drop=True)
    one_hot = pd.DataFrame(
        pt.nn.functional.one_hot(pt.tensor(df["label"])),
        columns=["a", "b", "c", "d", "e", "f", "g", "h"],
    )
    df = pd.merge(df, one_hot, left_index=True, right_index=True).drop(columns=["label"])
    print(df)


def validate(validation_split: float, use_cache: bool, max_length_secs: int, snr: float):
    correct_prediction = 0
    total_prediction = 0

    checkpoint = pt.load(TRAINED_MODEL_PATH)
    model = TramsAudioClassifier(ModelConfig())
    model.load_state_dict(checkpoint["state_dict"])

    dm = TramsDataModule(16, validation_split, max_length_secs, snr, use_cache=use_cache)
    dm.prepare_data()
    dm.setup()

    val_dataloader = dm.val_dataloader()

    model.eval()
    with pt.no_grad():
        for data in val_dataloader:
            inputs, labels = data["spectrogram"], data["label"]
            outputs = model(inputs)
            _, prediction = pt.max(outputs, 1)
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

    acc = correct_prediction / total_prediction
    print(f"Accuracy: {acc:.3f}, Total items: {total_prediction}")
