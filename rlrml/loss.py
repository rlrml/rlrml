import torch
import enum
import numpy as np

from scipy import stats


class LossType(enum.StrEnum):

    DIFFERENCE_WEIGHTED_MSE_LOSS = enum.auto()
    DIFFERENCE_AND_MSE_LOSS = enum.auto()
    WEIGHTED_MSE = enum.auto()

    def get_fn_from_args(self, **kwargs):
        if self == self.WEIGHTED_MSE:
            return WeightedLoss(weight_by=weight_by_mean_distance(**kwargs))
        elif self == self.DIFFERENCE_AND_MSE_LOSS:
            return difference_and_mse_loss(**kwargs)
        elif self == self.DIFFERENCE_WEIGHTED_MSE_LOSS:
            return WeightedLoss(weight_by=DifferenceLoss())


def difference_and_mse_loss(difference_scale=5.0, mse_scale=1.0):
    difference_scale = float(difference_scale)
    mse_scale = float(mse_scale)
    mse = torch.nn.MSELoss(reduction="none")

    def loss_fn(y_pred, y_train):
        mse_loss = mse_scale * mse(y_pred, y_train)
        diff_loss = difference_scale * DifferenceLoss()(y_pred, y_train)
        return (mse_loss + diff_loss)

    return loss_fn


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

    def __init__(self, base_loss=torch.nn.MSELoss(reduction='none')):
        super().__init__()
        self.base_loss = base_loss
        self.add_module("base_loss", self.base_loss)

    def forward(self, y_true, y_pred):
        # Compute all pairwise differences - ground truth and predictions
        y_true_diffs = y_true.unsqueeze(2) - y_true.unsqueeze(1)
        y_pred_diffs = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)

        # Calculate the difference loss as the Mean Absolute Error (MAE) between
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
        self.add_module("loss", self.loss_fn)
        self._weight_by = weight_by

    def forward(self, y_pred, y_true):
        weights = as_weight_matrix(self._weight_by(y_pred, y_true))

        # Ensure that input, target, and weights have the same shape
        assert y_pred.shape == y_true.shape == weights.shape, (
            "Shapes of input, target, and weights must be the same"
        )

        loss = self.loss_fn(y_pred, y_true)

        weights = weights.to(loss.device)
        weighted_loss = loss * weights

        return weighted_loss


def as_weight_matrix(tensor):
    num_elements = np.prod(tensor.shape)
    return num_elements * (tensor / tensor.sum())


def difference_weighted_mse():
    mse = torch.nn.MSELoss(reduction="none")
    diff_loss = DifferenceLoss()

    def loss(y_pred, y_train):
        weights = as_weight_matrix(diff_loss(y_pred, y_train))
        losses = weights * mse(y_pred, y_train)
        return losses

    return loss


def weight_by_mean_distance(
        target_distance=300, target_weight=3.0, min_weight=1.0, max_weight=20.0, use_tau=False
):
    scaling_factor = 2 * float(target_weight) / target_distance

    # Define the function to calculate weights
    def calculate_weights(labels, predictions):
        weights = []

        # Iterate over all samples in the batch
        for i in range(labels.size(0)):
            # Calculate the mean absolute deviation of the labels
            label = labels[i].cpu()
            prediction = predictions[i].cpu()
            mean = torch.mean(label)
            diff = torch.abs(label - mean)

            if use_tau:
                # Convert the regression values into rank-based permutations
                labels_rank = stats.rankdata(label.numpy())
                predictions_rank = stats.rankdata(prediction.numpy())

                # Calculate the tau distance between the labels and predictions
                tau, _ = stats.kendalltau(labels_rank, predictions_rank)

                # Scale the tau distance to be in the range [0, 1]
                tau = 0.5 * (tau + 1)
            else:
                tau = 0

            tau_weight = (1 + tau)

            # Incorporate the tau distance into the weight calculation
            weight = scaling_factor * diff * tau_weight

            # Clip the weights to the range [min_weight, max_weight]
            weight = torch.clamp(weight, min_weight, max_weight)

            weights.append(weight)

        return torch.stack(weights)

    return calculate_weights
