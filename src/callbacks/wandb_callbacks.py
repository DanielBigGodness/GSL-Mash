from typing import List, Any, Optional

import numpy as np
import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT
import time

from src.utils.metrics import Precision, NormalizedDCG


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if trainer.fast_dev_run:
        raise Exception(
            "Cannot use wandb callbacks since pytorch lightning disables loggers in `fast_dev_run=true` mode."
        )

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class WatchModel(Callback):
    """Make wandb watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log = log
        self.log_freq = log_freq

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(model=trainer.model, log="all", log_freq=self.log_freq)


class LogMetricsAndRunningTime(Callback):
    def __init__(self, top_k_list: List[int], device='cuda'):
        self.top_ks = top_k_list
        self.device = device
        self.preds = []
        self.targets = []
        self.training_epoch_times = []
        self.test_epoch_times = []

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.propensity_score = trainer.datamodule.propensity_score.to(self.device)

        if self.device == 'cuda':
            self.precisions = [torchmetrics.Precision(top_k=k).cuda() for k in self.top_ks]
            self.recalls = [torchmetrics.Recall(top_k=k).cuda() for k in self.top_ks]
            self.NDCGs = [torchmetrics.RetrievalNormalizedDCG(k=k).cuda() for k in self.top_ks]
            self.F1s = [torchmetrics.F1Score(top_k=k).cuda() for k in self.top_ks]
            self.PSPs = [Precision(top_k=k, propensity_score=self.propensity_score).cuda() for k in self.top_ks]
            self.PSDCGs = [NormalizedDCG(top_k=k, propensity_score=self.propensity_score).cuda() for k in self.top_ks]
            self.MRR = torchmetrics.RetrievalMRR().cuda()
        elif self.device == 'cpu':
            self.precisions = [torchmetrics.Precision(top_k=k) for k in self.top_ks]
            self.recalls = [torchmetrics.Recall(top_k=k) for k in self.top_ks]
            self.NDCGs = [torchmetrics.RetrievalNormalizedDCG(k=k) for k in self.top_ks]
            self.F1s = [torchmetrics.F1Score(top_k=k) for k in self.top_ks]
            self.PSPs = [Precision(top_k=k, propensity_score=self.propensity_score) for k in self.top_ks]
            self.PSDCGs = [NormalizedDCG(top_k=k, propensity_score=self.propensity_score) for k in self.top_ks]
            self.MRR = torchmetrics.RetrievalMRR()
        else:
            raise Exception('unknown device!')

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.training_epoch_start_time = time.time()

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", unused: Optional = None
    ) -> None:
        self.training_epoch_times.append(time.time() - self.training_epoch_start_time)

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.preds.append(outputs['preds'])
        self.targets.append(outputs['targets'])

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.test_epoch_start_time = time.time()

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.test_epoch_times.append(time.time() - self.test_epoch_start_time)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment
        log_data = {}
        for pred, target in zip(self.preds, self.targets):
            for p in self.precisions:
                p.update(pred, target)
            for r in self.recalls:
                r.update(pred, target)
            for n in self.NDCGs:
                n.update(pred, target, torch.tensor(range(pred.size(0))).unsqueeze(1).repeat(1, pred.size(1)))
            for f in self.F1s:
                f.update(pred, target)
            for p in self.PSPs:
                p.update(pred, target)
            for p in self.PSDCGs:
                p.update(pred, target)
            self.MRR.update(pred, target, torch.tensor(range(pred.size(0))).unsqueeze(1).repeat(1, pred.size(1)))
        for top_k, p, r, n, f, psp, psdcg in zip(self.top_ks, self.precisions, self.recalls, self.NDCGs, self.F1s, self.PSPs, self.PSDCGs):
            log_data[f'test/precision@{top_k}'] = p.compute()
            log_data[f'test/recall@{top_k}'] = r.compute()
            log_data[f'test/NDCG@{top_k}'] = n.compute()
            log_data[f'test/F1@{top_k}'] = f.compute()
            log_data[f'test/PSP@{top_k}'] = psp.compute()
            log_data[f'test/PSDCG@{top_k}'] = psdcg.compute()
        log_data['test/MRR'] = self.MRR.compute()

        log_data['epoch_time/training'] = np.mean(self.training_epoch_times)
        log_data['epoch_time/test'] = np.mean(self.test_epoch_times)

        experiment.log(log_data, commit=True)
