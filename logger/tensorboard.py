from datetime import datetime
from typing import Any, Callable

from torch.utils.tensorboard import SummaryWriter


class TensorboardWriter:
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.selected_module: str = ""

        self.step = 0
        self.mode = ""

        self.tb_writer_ftns = {
            "add_scalar",
            "add_scalars",
            "add_image",
            "add_images",
            "add_audio",
            "add_text",
            "add_histogram",
            "add_pr_curve",
            "add_embedding",
        }
        self.tag_mode_exceptions = {"add_histogram", "add_embedding"}
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        self.mode = mode
        previous_step = self.step
        self.step = step

        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar(
                "steps_per_sec",
                (self.step - previous_step) / duration.total_seconds(),
            )
            self.timer = datetime.now()

    def __getattr__(self, name: str) -> Callable:
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag: str, data: Any, *args, **kwargs) -> None:
                if add_data is not None:
                    try:
                        # add mode(train/valid) tag
                        if name not in self.tag_mode_exceptions:
                            tag = f"{tag}/{self.mode}"
                        add_data(tag, data, self.step, *args, **kwargs)
                    except Exception as e:
                        # Log error but don't crash training
                        print(f"Warning: Failed to log to tensorboard: {e}")

            return wrapper
        else:
            # Raise AttributeError for unknown attributes
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
