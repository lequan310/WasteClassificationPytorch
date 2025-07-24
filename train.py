import logging

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from data_loader.image_data_loader import get_dataloaders
from logger import setup_logging
from trainer import Trainer
from utils import create_saved_dirs, set_random_seed


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(config: DictConfig):
    paths = create_saved_dirs(config)
    setup_logging(save_dir=paths["log_dir"])
    set_random_seed(config.trainer.seed)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    dataloaders = get_dataloaders(config.dataloaders.train)

    model: torch.nn.Module = instantiate(config.model).to(device)
    logger.info(model)

    criterion = instantiate(config.loss).to(device)
    metrics = instantiate(config.metrics.evaluation)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)
    writer = instantiate(config.writer, log_dir=paths["log_dir"])
    config.trainer.save_dir = paths["save_dir"]

    trainer = Trainer(
        model=model,
        criterion=criterion,
        metrics=metrics,
        optimizer=optimizer,
        config=config,
        device=device,
        data_loader=dataloaders["train"],
        valid_data_loader=dataloaders.get("val"),
        lr_scheduler=lr_scheduler,
        writer=writer,
        logger=logger,
    )

    trainer.train()


if __name__ == "__main__":
    main()
