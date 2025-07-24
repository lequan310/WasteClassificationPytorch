import logging

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from data_loader.image_data_loader import get_test_dataloader
from logger import setup_logging
from trainer import Inferencer
from utils import create_saved_dirs, set_random_seed


@hydra.main(version_base=None, config_path="configs", config_name="test")
def main(config: DictConfig):
    paths = create_saved_dirs(config, mode="test")
    setup_logging(save_dir=paths["log_dir"])
    set_random_seed(config.inferencer.seed)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    data_loader = get_test_dataloader(config.dataloaders.test)

    model = instantiate(config.model).to(device)
    logger.info(model)

    criterion = instantiate(config.loss).to(device)
    metrics = instantiate(config.metrics.evaluation)

    inferencer = Inferencer(
        model=model,
        criterion=criterion,
        metrics=metrics,
        device=device,
        data_loader=data_loader,
        config=config,
        logger=logger,
    )

    inferencer.evaluate()


if __name__ == "__main__":
    main()
