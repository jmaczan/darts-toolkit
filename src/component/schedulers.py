class TemperatureScheduler:
    def __init__(
        self,
        temperature_start: float = 1.0,
        temperature_end: float = 0.1,
        epochs: int = 50,
    ):
        self.temperature_start = temperature_start
        self.temperature_end = temperature_end
        self.epochs = epochs

    def __call__(self, epoch: int) -> float:
        if epoch < 0 or self.epochs <= 0:
            return self.temperature_start
        if epoch >= self.epochs:
            return self.temperature_end

        return (
            self.temperature_start
            - (self.temperature_start - self.temperature_end) * epoch / self.epochs
        )


class DropPathScheduler:
    def __init__(
        self, drop_path_prob_start: float, drop_path_prob_end: float, epochs: int
    ):
        self.drop_path_prob_start = drop_path_prob_start
        self.drop_path_prob_end = drop_path_prob_end
        self.epochs = epochs

    def __call__(self, epoch: int) -> float:
        if epoch < 0 or self.epochs <= 0:
            return self.drop_path_prob_start
        if epoch >= self.epochs:
            return self.drop_path_prob_end

        return (
            self.drop_path_prob_start
            + (self.drop_path_prob_end - self.drop_path_prob_start)
            * epoch
            / self.epochs
        )
