# early stopping.py

class EarlyStopping:
    """
    Implements early stopping to halt training when validation loss stops improving.
    
    Parameters:
        patience (int): Number of epochs to wait after last improvement before stopping.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
    
    Attributes:
        best_score (float): Best validation score seen so far.
        best_model_state (dict): State dict of the model with the best score.
    """
    def __init__(self, patience=5, delta=0):
        """
        Args:
            patience (int): Number of epochs to wait before stopping if no improvement.
            delta (float): Minimum change to qualify as an improvement.
        """
        self.patience = patience  # Number of epochs to wait before stopping
        self.delta = delta        # Minimum change to qualify as improvement
        self.best_score = None   # Best validation score seen so far
        self.early_stop = False  # Flag to indicate if training should stop
        self.counter = 0         # Counts epochs with no improvement
        self.best_model_state = None  # Stores best model state
        self.best_epoch = None   # Stores the epoch at which the best model was found

    def __call__(self, val_loss, model, epoch=None):
        """
        Call method to update early stopping logic.
        Args:
            val_loss (float): Current validation loss.
            model (torch.nn.Module): Model being trained.
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
            if epoch is not None:
                self.best_epoch = epoch
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0
            if epoch is not None:
                self.best_epoch = epoch
    def get_best_epoch(self):
        """
        Returns the epoch at which the best model was found.
        Returns:
            int or None: Epoch number or None if not set.
        """
        return self.best_epoch

    def load_best_model(self, model):
        """
        Loads the best model state into the given model.
        Args:
            model (torch.nn.Module): Model to load the best state into.
        """
        model.load_state_dict(self.best_model_state)