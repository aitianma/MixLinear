"""
GluonTS integration for MixLinear, following the same pattern GluonTS itself
uses for SimpleFeedForwardEstimator (see gluonts.torch.model.simple_feedforward):
a plain nn.Module -> a LightningModule that adds training/validation steps ->
a PyTorchLightningEstimator that wires up transformation / data loaders /
predictor construction.

MixLinear is deterministic (a single point forecast per series, no learned
distribution), so we treat its output as a 1-sample SampleForecast, which is
what GluonTS's `evaluate_model` / the leaderboard's metrics expect for point
forecasters (mean and median of a 1-sample forecast are just that sample).

MixLinear is also channel-independent (all its weights are shared across the
channel dimension; there is no cross-channel mixing), so multivariate
datasets are converted to independent univariate series
(`Dataset(..., to_univariate=True)`) upstream -- this is mathematically
equivalent to running the model jointly over all channels and matches the
convention used by the other simple baselines in gift-eval (e.g. naive,
feedforward).
"""
from typing import Any, Dict, Iterable, List, Optional

import lightning.pytorch as pl
import torch
import torch.nn as nn

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset as GluonTSDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import as_stacked_batches
from gluonts.itertools import Cyclic
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.model.forecast_generator import SampleForecastGenerator
from gluonts.transform import (
    AddObservedValuesIndicator,
    ExpectedNumInstanceSampler,
    InstanceSampler,
    InstanceSplitter,
    SelectFields,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
)

from model import MixLinearCore

PREDICTION_INPUT_NAMES = ["past_target"]
TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]


class MixLinearPredictionNetwork(nn.Module):
    """Adapts MixLinearCore (batch, seq, channel) to GluonTS's flat
    (batch, time) univariate convention, and adds a sample-dimension of 1
    so it can be consumed by SampleForecastGenerator."""

    def __init__(self, prediction_length: int, context_length: int, period_len: int, lpf: int, alpha: float):
        super().__init__()
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.core = MixLinearCore(
            seq_len=context_length,
            pred_len=prediction_length,
            enc_in=1,
            period_len=period_len,
            lpf=lpf,
            alpha=alpha,
        )

    def forward(self, past_target: torch.Tensor) -> torch.Tensor:
        # past_target: (batch, context_length) -> (batch, context_length, 1)
        x = past_target.unsqueeze(-1)
        y = self.core(x)  # (batch, pred_len, 1)
        y = y.squeeze(-1)  # (batch, pred_len)
        return y.unsqueeze(1)  # (batch, num_samples=1, pred_len)


class MixLinearLightningModule(pl.LightningModule):
    @validated()
    def __init__(self, model_kwargs: dict, lr: float = 1e-2, weight_decay: float = 1e-8):
        super().__init__()
        self.save_hyperparameters()
        self.model = MixLinearPredictionNetwork(**model_kwargs)
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = nn.MSELoss(reduction="none")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _compute_loss(self, batch):
        forecast = self.model(past_target=batch["past_target"])  # (batch, 1, pred_len)
        forecast = forecast.squeeze(1)  # (batch, pred_len)
        target = batch["future_target"]
        observed = batch["future_observed_values"]
        loss = self.loss_fn(forecast, target) * observed
        denom = observed.sum().clamp_min(1.0)
        return loss.sum() / denom

    def training_step(self, batch, batch_idx: int):
        loss = self._compute_loss(batch)
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        loss = self._compute_loss(batch)
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)


class MixLinearEstimator(PyTorchLightningEstimator):
    @validated()
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        period_len: int,
        lpf: int,
        alpha: float = 0.5,
        lr: float = 1e-2,
        weight_decay: float = 1e-8,
        batch_size: int = 64,
        num_batches_per_epoch: int = 100,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
    ) -> None:
        default_trainer_kwargs = {"max_epochs": 30, "gradient_clip_val": 1.0}
        if trainer_kwargs is not None:
            default_trainer_kwargs.update(trainer_kwargs)
        super().__init__(trainer_kwargs=default_trainer_kwargs)

        self.prediction_length = prediction_length
        self.context_length = context_length
        self.period_len = period_len
        self.lpf = lpf
        self.alpha = alpha
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch

        self.train_sampler = train_sampler or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=prediction_length
        )
        self.validation_sampler = validation_sampler or ValidationSplitSampler(min_future=prediction_length)

    def create_transformation(self) -> Transformation:
        return SelectFields(
            [FieldName.ITEM_ID, FieldName.INFO, FieldName.START, FieldName.TARGET],
            allow_missing=True,
        ) + AddObservedValuesIndicator(
            target_field=FieldName.TARGET,
            output_field=FieldName.OBSERVED_VALUES,
        )

    def create_lightning_module(self) -> pl.LightningModule:
        return MixLinearLightningModule(
            lr=self.lr,
            weight_decay=self.weight_decay,
            model_kwargs={
                "prediction_length": self.prediction_length,
                "context_length": self.context_length,
                "period_len": self.period_len,
                "lpf": self.lpf,
                "alpha": self.alpha,
            },
        )

    def _create_instance_splitter(self, module: MixLinearLightningModule, mode: str):
        assert mode in ["training", "validation", "test"]
        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            time_series_fields=[FieldName.OBSERVED_VALUES],
            dummy_value=0.0,
        )

    def create_training_data_loader(self, data: GluonTSDataset, module: MixLinearLightningModule, shuffle_buffer_length: Optional[int] = None, **kwargs) -> Iterable:
        data = Cyclic(data).stream()
        instances = self._create_instance_splitter(module, "training").apply(data, is_train=True)
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            shuffle_buffer_length=shuffle_buffer_length,
            field_names=TRAINING_INPUT_NAMES,
            output_type=torch.tensor,
            num_batches_per_epoch=self.num_batches_per_epoch,
        )

    def create_validation_data_loader(self, data: GluonTSDataset, module: MixLinearLightningModule, **kwargs) -> Iterable:
        instances = self._create_instance_splitter(module, "validation").apply(data, is_train=True)
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            field_names=TRAINING_INPUT_NAMES,
            output_type=torch.tensor,
        )

    def create_predictor(self, transformation: Transformation, module: MixLinearLightningModule) -> PyTorchPredictor:
        prediction_splitter = self._create_instance_splitter(module, "test")
        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=PREDICTION_INPUT_NAMES,
            prediction_net=module,
            forecast_generator=SampleForecastGenerator(),
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            device="auto",
        )
