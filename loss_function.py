import abc
import torch
from torch.nn import functional as F


class LossFunction(abc.ABC):
    """The interface for loss functions that the CachePolicyModel can use."""

    @abc.abstractmethod
    def __call__(self, predicted_log_reuse_distances, true_log_reuse_distances, mask) -> torch.FloatTensor:
        """Computes the value of the loss function.
        
        Args:
            predicted_log_reuse_distances (torch.FloatTensor): the log of the predicted reuse distances
              with the shape of (batch_size, num_lines).
            true_log_reuse_distances (torch.FloatTensor): the log of the true reuse distances with
                the shape of (batch_size, num_lines).
            mask (torch.ByteTensor): masks out elements if the value is 0 with the shape of
                (batch_size, num_lines).

        Returns:
            torch.FloatTensor: the value of the loss function with the shape of (batch_size,).
        """

class ReuseDistanceLoss(LossFunction):
    """Computes the mean squared error (MSE) between the predicted and true log reuse distances."""

    def __init__(self):
        super().__init__()
        print("Expects that all calls to loss are labeled with Belady's optimal policy.")

    def __call__(self, predicted_log_reuse_distances, true_log_reuse_distances, mask) -> torch.FloatTensor:
        """Computes the MSE between the predicted and true log reuse distances."""
        return F.mse_loss(predicted_log_reuse_distances * mask.float(),
                          true_log_reuse_distances * mask.float(),
                          reduction='none').mean(dim=-1)