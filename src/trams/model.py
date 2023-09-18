import os
from dataclasses import dataclass
from typing import Literal, Any

import torch as pt
from torch import nn
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import Accuracy


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
        x = pt.unsqueeze(inputs["spectrogram"], 1)
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
        self.learning_rate = config.learning_rate
        self.model = AudioClassifier(config)
        self.save_hyperparameters()
        self.metrics = {"acc": Accuracy("multiclass", num_classes=config.num_classes)}

    def forward(self, batch: pt.Tensor) -> pt.Tensor:
        return self.model(batch)

    def training_step(self, batch: dict[str, Any], _: int) -> STEP_OUTPUT:
        logits = self.model(batch)
        labels = batch["label"]
        loss = nn.functional.cross_entropy(logits, labels)
        _, preds = pt.max(logits, 1)
        acc = self.metrics["acc"](preds, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch: dict[str, Any], _: int) -> STEP_OUTPUT:
        logits = self.model(batch)
        labels = batch["label"]
        loss = nn.functional.cross_entropy(logits, labels)
        _, preds = pt.max(logits, 1)
        acc = self.metrics["acc"](preds, labels)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, logger=True)
        return loss

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
