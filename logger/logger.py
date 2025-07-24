import logging
import logging.config
from pathlib import Path
from typing import Union

from utils import read_json


def setup_logging(
    save_dir: Union[str, Path],
    log_config: str = "logger/logging_config.json",
    default_level: int = logging.INFO,
) -> None:
    """Setup logging configuration"""
    log_config_path = Path(log_config)
    if log_config_path.is_file():
        try:
            config = read_json(log_config_path)
            # modify logging paths based on run config
            for _, handler in config["handlers"].items():
                if "filename" in handler:
                    handler["filename"] = str(Path(save_dir) / handler["filename"])

            logging.config.dictConfig(config)
        except (KeyError, ValueError) as e:
            print(f"Error parsing logging config: {e}")
            logging.basicConfig(level=default_level)
    else:
        print(f"Warning: logging configuration file is not found in {log_config_path}.")
        logging.basicConfig(level=default_level)
