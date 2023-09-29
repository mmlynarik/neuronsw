import os
from dataclasses import dataclass
from typing import Literal, Any

import wandb
import torch as pt
import matplotlib.pyplot as plt
from torch import nn
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix

from trams.config import LABELS_NAMES_SHORT
from trams.utils import plot_confusion_matrix


Device = Literal["cuda", "cpu"]


@dataclass
class ModelConfig:
    conv_size_base: int = 8
    conv_size_multiplier: int = 2
    num_blocks: int = 4
    kernel_size: int = 3
    stride: int = 2
    padding: int = 1
    kaiming_init_a: float = 0.1
    num_channels: int = 1
    num_classes: int = 9
    learning_rate: float = 0.001
    device: Device = "cuda"


def remove_config_yaml():
    """If not deleted, it would interfere with next runs."""
    if os.path.exists("config.yaml"):
        os.remove("config.yaml")


class ConvolutionBlock(nn.Module):
    def __init__(self, config: ModelConfig, block_id: int):
        super().__init__()
        in_channels = (
            config.num_channels
            if block_id == 0
            else config.conv_size_base * (config.conv_size_multiplier ** (block_id - 1))
        )
        out_channels = config.conv_size_base * (config.conv_size_multiplier**block_id)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding,
        )
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_channels)
        nn.init.kaiming_normal_(self.conv.weight, a=config.kaiming_init_a)
        self.conv.bias.data.zero_()

    def forward(self, x: pt.Tensor) -> pt.Tensor:
        x = self.conv(x)
        x = self.relu(x)
        x = self.norm(x)
        return x


class AudioClassifier(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.convolutions = nn.ModuleList([ConvolutionBlock(config, idx) for idx in range(config.num_blocks)])
        self.average_pooling = nn.AdaptiveAvgPool2d(output_size=1)
        dense_in_features = config.conv_size_base * (config.conv_size_multiplier ** (config.num_blocks - 1))
        self.dense = nn.Linear(in_features=dense_in_features, out_features=config.num_classes)

    def forward(self, inputs: dict[str, Any]) -> pt.Tensor:
        x = pt.unsqueeze(inputs, 1)
        for block in self.convolutions:
            x = block(x)
        x = self.average_pooling(x)
        x = x.view(x.size()[0], -1)
        x = self.dense(x)
        return x


class TramsAudioClassifier(LightningModule):
    """Pytorch-Lightning encapsulation of audio classification model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        num_classes = config.num_classes
        self.learning_rate = config.learning_rate
        self.model = AudioClassifier(config)
        self.metrics = nn.ModuleDict(
            {
                "train_accuracy": MulticlassAccuracy(num_classes),
                "val_accuracy": MulticlassAccuracy(num_classes),
            }
        )
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes)
        self.save_hyperparameters()

    def forward(self, batch: pt.Tensor) -> pt.Tensor:
        return self.model(batch)

    def log_metrics(self, stage: str, preds: pt.Tensor, labels: pt.Tensor, on_step: bool, on_epoch: bool):
        """Log scalar-valued metrics into logger and progress bar."""
        for name, metric in self.metrics.items():
            if stage in name:
                metric(preds, labels)
                self.log(name, value=metric, on_step=on_step, on_epoch=on_epoch)

    def training_step(self, batch: dict[str, Any], _: int) -> STEP_OUTPUT:
        logits = self.model(batch["spectrogram"])
        labels = batch["label"]
        loss = nn.functional.cross_entropy(logits, labels)
        _, preds = pt.max(logits, dim=1)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_metrics("train", preds, labels, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: dict[str, Any], _: int) -> STEP_OUTPUT:
        logits = self.model(batch["spectrogram"])
        labels = batch["label"]
        loss = nn.functional.cross_entropy(logits, labels)
        _, preds = pt.max(logits, 1)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_metrics("val", preds, labels, on_step=True, on_epoch=True)
        self.confusion_matrix.update(preds, labels)
        return loss

    def on_validation_epoch_end(self):
        confusion_matrix = self.confusion_matrix.compute()
        fig, ax = plt.subplots()
        plot_confusion_matrix(confusion_matrix, ax=ax, labels=LABELS_NAMES_SHORT)
        wandb.log({"plot": plt, "trainer/global_step": self.trainer.current_epoch})
        plt.close(fig)
        self.confusion_matrix.reset()

    def test_step(self, batch: dict[str, Any], _: int) -> STEP_OUTPUT:
        pass

    def configure_optimizers(self) -> Any:
        optimizer = pt.optim.Adam(self.parameters(), self.learning_rate)
        return optimizer

    def on_fit_start(self) -> None:
        logger: WandbLogger = self.logger
        logger.experiment.log_code(
            root="./src",
            name=f"source-code-{logger.experiment.id}",
            include_fn=lambda path: path.endswith(".py") or path.endswith(".yaml") or path.endswith(".json"),
        )

    def on_validation_start(self) -> None:
        """This will affect both fit and validation stage as validation is run also during training."""
        remove_config_yaml()
