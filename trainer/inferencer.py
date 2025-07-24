import csv
import logging

import torch
from torch import nn
from tqdm import tqdm

from base import BaseMetric, BaseTrainer
from utils import MetricTracker


class Inferencer(BaseTrainer):
    """Class for evaluating a model on a dataset."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        metrics: list[BaseMetric],
        config: dict,
        device: torch.device,
        logger: logging.Logger,
        data_loader: torch.utils.data.DataLoader,
    ):
        self.config = config
        self.cfg_trainer = self.config.inferencer
        self.logger = logger

        self.device = device
        self.model = model
        self.data_loader = data_loader

        self.criterion = criterion
        self.metrics = metrics
        # Check if metrics are provided before creating the tracker
        if self.metrics:
            self.evaluation_metrics = MetricTracker(
                "loss",
                *[m.name for m in self.metrics],
                writer=None,
            )
        else:
            # If no metrics, just track loss
            self.evaluation_metrics = MetricTracker("loss", writer=None)

        weight_path = self.cfg_trainer.get("weight_path", None)
        if weight_path is not None:
            self._load_weights(weight_path)

        self.output_file = self.cfg_trainer.get("output_file", None)
        if self.output_file:
            self.logger.info(f"Predictions will be saved to {self.output_file}")

    def _load_weights(self, weight_path: str):
        """
        Load model weights from the specified path.
        """
        self.logger.info(f"Loading weights from {weight_path}")
        checkpoint = torch.load(
            weight_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.logger.info("Weights loaded successfully.")

    def evaluate(self):
        self.model.eval()
        self.evaluation_metrics.reset()

        ### ADDED ###
        # 1. Reset the state of all metric objects before starting
        if self.metrics:
            for met in self.metrics:
                met.reset()

        output_file_handler = None
        csv_writer = None
        if self.output_file:
            output_file_handler = open(self.output_file, "w", newline="")
            csv_writer = csv.writer(output_file_handler)
            csv_writer.writerow(["prediction", "target"])

        try:
            # Using inference_mode is slightly more efficient than no_grad()
            with torch.inference_mode():
                for batch_idx, (data, target) in enumerate(tqdm(self.data_loader)):
                    data, target = data.to(self.device), target.to(self.device)

                    output = self.model(data)
                    loss = self.criterion(output, target)

                    self.evaluation_metrics.update("loss", loss.item())

                    ### CHANGED ###
                    # 2. Update the state of each metric with the current batch
                    if self.metrics:
                        for met in self.metrics:
                            met.update(output, target)

                    if csv_writer:
                        preds = torch.argmax(output, dim=-1)
                        preds_list = preds.cpu().tolist()
                        targets_list = target.cpu().tolist()
                        for p, t in zip(preds_list, targets_list):
                            csv_writer.writerow([p, t])
        finally:
            if output_file_handler:
                output_file_handler.close()

        ### ADDED ###
        # 3. Compute the final score from all accumulated batches
        if self.metrics:
            for met in self.metrics:
                self.evaluation_metrics.update(met.name, met.compute())

        log = self.evaluation_metrics.result()
        self.logger.info("--- Evaluation Results ---")
        for key, value in log.items():
            self.logger.info("    {:15s}: {}".format(str(key), value))

        return log
