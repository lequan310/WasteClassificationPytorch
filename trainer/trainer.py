import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torchvision.utils import make_grid

from base import BaseMetric, BaseTrainer
from logger import TensorboardWriter
from utils import MetricTracker, inf_loop


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        metrics: list[BaseMetric],
        optimizer: Optimizer,
        config: dict,
        device: torch.device,
        logger: logging.Logger,
        data_loader: torch.utils.data.DataLoader,
        writer: TensorboardWriter,
        valid_data_loader: Optional[torch.utils.data.DataLoader] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        len_epoch=None,
    ):
        super().__init__(
            model=model,
            criterion=criterion,
            metrics=metrics,
            optimizer=optimizer,
            config=config,
            writer=writer,
            logger=logger,
        )

        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            self.len_epoch = len(self.data_loader)
        else:
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        # MetricTracker will now store the final computed scores at the end of the epoch
        self.train_metrics = MetricTracker(
            "loss", *[m.name for m in self.metrics], writer=self.writer
        )
        self.valid_metrics = MetricTracker(
            "loss", *[m.name for m in self.metrics], writer=self.writer
        )

    def _train_epoch(self, epoch: int):
        """
        Training logic for an epoch
        """
        self.model.train()
        self.train_metrics.reset()

        ### ADDED ###
        # Reset all metric objects at the start of the training epoch
        for met in self.metrics:
            met.reset()

        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update("loss", loss.item())

            ### CHANGED ###
            # Update the state of each metric object. Don't compute the value here.
            for met in self.metrics:
                met.update(output, target)

            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), loss.item()
                    )
                )
                self.writer.add_image(
                    "input", make_grid(data.cpu(), nrow=8, normalize=True)
                )

            if batch_idx == self.len_epoch:
                break

        ### ADDED ###
        # Now compute the final metrics for the entire training epoch
        for met in self.metrics:
            self.train_metrics.update(met.name, met.compute())

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch: int):
        """
        Validate after training an epoch
        """
        self.model.eval()
        self.valid_metrics.reset()

        ### ADDED ###
        # Reset all metric objects at the start of the validation epoch
        for met in self.metrics:
            met.reset()

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step(
                    (epoch - 1) * len(self.valid_data_loader) + batch_idx, "valid"
                )
                self.valid_metrics.update("loss", loss.item())

                ### CHANGED ###
                # Update the state of each metric object
                for met in self.metrics:
                    met.update(output, target)

                self.writer.add_image(
                    "input", make_grid(data.cpu(), nrow=8, normalize=True)
                )

        ### ADDED ###
        # Now compute the final metrics for the entire validation dataset
        for met in self.metrics:
            self.valid_metrics.update(met.name, met.compute())

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")

        return self.valid_metrics.result()

    def _progress(self, batch_idx: int):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
