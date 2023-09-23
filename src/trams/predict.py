import argparse
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
    TRAMS_LABELS_NAMES,
)
from trams.model import TramsAudioClassifier, ModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-wav", type=str, help="Input wav file path")
    parser.add_argument("--output-csv", type=str, help="Output csv file")
    args = parser.parse_args()
    return args


def get_output_mask(predictions: list[tuple[float, int]]) -> list[bool]:
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
    return output_mask


def predict_trams_from_wav(input_wav: Path, output_csv: Path):
    audio, sample_rate = torchaudio.load(input_wav)
    audio_metadata: AudioMetaData = torchaudio.info(input_wav)
    if sample_rate != 22050 or audio_metadata.num_channels != 1:
        raise ValueError("Current model only supports sample rate of 22050 and mono channel.")

    num_frames = audio_metadata.num_frames
    num_seconds = num_frames / sample_rate
    print(f"Loaded {Path(input_wav).parts[-1]} file spanning {num_seconds} seconds.")

    mel_spectrogram = transforms.MelSpectrogram(sample_rate, n_fft=NUM_FFT, n_mels=NUM_MELS)
    amplitude_transformer = transforms.AmplitudeToDB(top_db=MAX_DB)

    slice_frames = int(MAX_LENGTH_SECS * sample_rate)
    offset_frames = int(ONE_TENTH_SEC * sample_rate)

    model = TramsAudioClassifier.load_from_checkpoint(TRAINED_MODEL_PATH)

    predictions = []
    model.to(device="cpu").eval()
    with pt.no_grad():
        position = 0
        while position <= num_frames - slice_frames:
            slice = audio[:, position : position + slice_frames]
            spectrogram: pt.Tensor = amplitude_transformer(mel_spectrogram(slice))
            output = model(spectrogram)
            prediction = pt.argmax(output.softmax(dim=1), dim=1).item()
            if prediction == model.hparams.config.num_classes - 1:
                position += offset_frames
            else:
                predictions.append((position / sample_rate, prediction))
                position += slice_frames

    output_mask = get_output_mask(predictions)
    df = pd.DataFrame(predictions, columns=["seconds_offset", "label"])[output_mask].reset_index(drop=True)
    one_hot = pd.DataFrame(pt.nn.functional.one_hot(pt.tensor(df["label"])),columns=TRAMS_LABELS_NAMES)
    df = pd.merge(df, one_hot, left_index=True, right_index=True).drop(columns=["label"])
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    args = parse_args()
    predict_trams_from_wav(args.input_wav, args.output_csv)
