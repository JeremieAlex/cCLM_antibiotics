import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop when the validation loss has not improved by ``threshold`` for ``patience`` epochs.

    It only decides when to stop; the saved model is always the last epoch's.
    """

    def __init__(self, patience: int = 5, threshold: float = 0.05):
        self.patience = patience
        self.threshold = threshold
        self.counter = 0
        self.best_loss = None
        self.best_epoch = 0
        self.stop = False

    def __call__(self, val_loss: float, epoch: int) -> None:
        if self.best_loss is None or val_loss < self.best_loss - self.threshold:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            return
        self.counter += 1
        logger.info(f"No improvement for {self.counter}/{self.patience} epochs (best {self.best_loss:.4f})")
        if self.counter >= self.patience:
            self.stop = True
