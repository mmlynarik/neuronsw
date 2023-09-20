from pathlib import Path

import torchaudio
import torch
import pandas as pd
from torchaudio.backend.common import AudioMetaData
from torchaudio import transforms


from trams.config import (
    NUM_FFT,
    NUM_MELS,
    MAX_DB,
    MAX_LENGTH_SECS,
    ONE_TENTH_SEC,
    TRAINDED_MODEL_PATH,
)
from trams.model import TramsAudioClassifier, ModelConfig


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
    slices = int((num_frames - slice_frames) / offset_frames)

    checkpoint = torch.load(TRAINDED_MODEL_PATH)
    model = TramsAudioClassifier(ModelConfig())
    model.load_state_dict(checkpoint["state_dict"])

    predictions = []
    with torch.no_grad():
        for i in range(slices):
            slice = audio[:, i * offset_frames : i * offset_frames + slice_frames]
            spectrogram = amplitude_transformer(mel_spectrogram(slice))
            output: torch.Tensor = model(spectrogram)
            prediction = torch.argmax(output.softmax(dim=1), dim=1).item()
            predictions.append(prediction)

    print(pd.Series(predictions).value_counts())
