import pandas as pd


class MetricTracker:
    """Track and compute running averages of metrics."""

    def __init__(self, *keys: str, writer=None):
        """
        Args:
            *keys (list[str]): list (as positional arguments) of metric
                names (may include the names of losses)
            writer: experiment tracker.
                Not used in this code version. Can be used to log metrics
                from each batch.
        """
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self) -> None:
        """Reset all tracked metrics."""
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key: str, value: float, n: int = 1) -> None:
        """Update metric with new value."""
        if key not in self._data.index:
            raise KeyError(f"Key '{key}' not found in tracked metrics")

        if n <= 0:
            raise ValueError("Number of samples 'n' must be positive")

        if self.writer is not None:
            self.writer.add_scalar(key, value)

        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n

        if self._data.loc[key, "counts"] > 0:
            self._data.loc[key, "average"] = (
                self._data.loc[key, "total"] / self._data.loc[key, "counts"]
            )
        else:
            self._data.loc[key, "average"] = 0.0

    def avg(self, key: str) -> float:
        """Get average value for a specific key."""
        return self._data.loc[key, "average"]

    def result(self) -> dict[str, float]:
        """Get dictionary of all average values."""
        return self._data["average"].to_dict()

    def keys(self) -> pd.Index:
        """
        Return all metric names defined in the MetricTracker.

        Returns:
            metric_keys (Index): all metric names in the table.
        """
        return self._data.index
