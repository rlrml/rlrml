import torch
import enum
import numpy as np

from inspect import signature


def as_weight_matrix(tensor):
    num_elements = np.prod(tensor.shape)
    return num_elements * (tensor / tensor.sum())


def loss_takes_mask(loss_fn):
    target = loss_fn
    if hasattr(loss_fn, 'forward'):
        target = loss_fn.forward
    return len(signature(target).parameters) >= 3


class LossType(enum.StrEnum):

    DIFFERENCE_WEIGHTED_MSE_LOSS = enum.auto()
    DIFFERENCE_AND_MSE_LOSS = enum.auto()
    WEIGHTED_MSE = enum.auto()
    MSE = enum.auto()

    def get_fn_from_args(self, **kwargs):
        if self == self.WEIGHTED_MSE:
            return WeightedLoss(weight_by=weight_by_mean_distance(**kwargs))
        elif self == self.DIFFERENCE_AND_MSE_LOSS:
            return difference_and_mse_loss(**kwargs)
        elif self == self.DIFFERENCE_WEIGHTED_MSE_LOSS:
            return WeightedLoss(weight_by=DifferenceLoss())
        elif self == self.MSE:
            return torch.nn.MSELoss(reduction='none')


class CombinedLoss(torch.nn.Module):
    def __init__(
            self,
            left_loss,
            right_loss,
            left_scale=1.0,
            right_scale=1.0,
            left_takes_loss=None,
            right_takes_loss=None,
    ):
        super().__init__()
        self.left_loss = left_loss
        self.right_loss = right_loss
        self.left_scale = left_scale
        self.right_scale = right_scale
        self.add_module("left_loss", self.left_loss)
        self.add_module("right_loss", self.right_loss)
        self.left_takes_mask = left_takes_loss is not None or loss_takes_mask(self.left_loss)
        self.right_takes_mask = right_takes_loss is not None or loss_takes_mask(self.right_loss)

    def forward(self, y_pred, y_true, mask=None):
        mask = mask if mask is not None else torch.ones_like(y_true)
        left_loss = (
            self.left_loss(y_pred, y_true, mask=mask)
            if self.left_takes_mask
            else self.left_loss(y_pred, y_true)
        )
        right_loss = (
            self.right_loss(y_pred, y_true, mask=mask)
            if self.right_takes_mask
            else self.right_loss(y_pred, y_true)
        )
        return self.left_scale * left_loss + self.right_scale * right_loss


def difference_and_mse_loss(difference_scale=5.0, mse_scale=1.0):
    return CombinedLoss(
        DifferenceLoss(), torch.nn.MSELoss(reduction='none'),
        left_scale=difference_scale, right_scale=mse_scale,
    )


class ProportionalLoss(torch.nn.Module):
    def __init__(self, base_loss=torch.nn.MSELoss(reduction='none')):
        super().__init__()
        self.base_loss = base_loss
        self.add_module("base_loss", self.base_loss)

    def forward(self, y_pred, y_true):
        base_loss = self.base_loss(y_pred, y_true)
        weights = torch.abs(y_true)
        return torch.mean(base_loss / (weights + 1e-7))


class DifferenceLoss(torch.nn.Module):
    """Compute a loss defined by the relative differences of continuous values.

    The idea behind this loss function is to encourage a model to understand the
    relative ordering between different predictions made on a specific sample.

    """

    def __init__(self, base_loss=torch.nn.MSELoss(reduction='none')):
        super().__init__()
        self.base_loss = base_loss
        self.add_module("base_loss", self.base_loss)

    def forward(self, y_true, y_pred, mask=None):
        mask = mask if mask is not None else torch.ones_like(y_true)
        # Expand the mask for the pairwise differences operation
        mask = mask.unsqueeze(2) * mask.unsqueeze(1)

        # Compute all pairwise differences - ground truth and predictions
        y_true_diffs = y_true.unsqueeze(2) - y_true.unsqueeze(1)
        y_pred_diffs = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)

        # Apply the mask to the diffs, set unmasked values to some neutral value, e.g. 0
        y_true_diffs = y_true_diffs * mask
        y_pred_diffs = y_pred_diffs * mask

        # Calculate the difference loss as the base loss (defaults to MSE) between
        # actual and predicted differences
        diff_loss = torch.mean(self.base_loss(y_pred_diffs, y_true_diffs), dim=2)

        return diff_loss


class WeightedLoss(torch.nn.Module):
    def __init__(
            self,
            weight_by=DifferenceLoss(),
            loss_fn=torch.nn.MSELoss(reduction="none")
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.weight_by = weight_by
        self.loss_takes_mask = loss_takes_mask(self.loss_fn)
        self.weight_takes_mask = loss_takes_mask(self.weight_by)

    def forward(self, y_pred, y_true, mask=None):
        mask = mask if mask is not None else torch.ones_like(y_true)
        weights = (
            self.weight_by(y_pred, y_true, mask=mask)
            if self.weight_takes_mask
            else self.weight_by(y_pred, y_true)
        )
        weights = as_weight_matrix(weights)

        # Ensure that input, target, and weights have the same shape
        assert y_pred.shape == y_true.shape == weights.shape, (
            "Shapes of input, target, and weights must be the same"
        )

        loss = (
            self.loss_fn(y_pred, y_true, mask=mask)
            if self.loss_takes_mask
            else self.loss_fn(y_pred, y_true)
        )

        weights = weights.to(loss.device)
        weighted_loss = loss * weights

        return weighted_loss


def weight_by_mean_distance(
        target_distance=300, target_weight=3.0, min_weight=1.0, max_weight=20.0
):
    scaling_factor = 2 * float(target_weight) / target_distance

    # Define the function to calculate weights
    def calculate_weights(labels, predictions):
        weights = []

        # Iterate over all samples in the batch
        for i in range(labels.size(0)):
            # Calculate the mean absolute deviation of the labels
            label = labels[i].cpu()
            mean = torch.mean(label)
            diff = torch.abs(label - mean)

            # Incorporate the tau distance into the weight calculation
            weight = scaling_factor * diff

            # Clip the weights to the range [min_weight, max_weight]
            weight = torch.clamp(weight, min_weight, max_weight)

            weights.append(weight)

        return torch.stack(weights)

    return calculate_weights
