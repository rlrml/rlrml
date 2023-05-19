import torch
import numpy as np

from scipy import stats


class WeightedByMSELoss(torch.nn.Module):
    def __init__(self, weight_by=None):
        super().__init__()
        self.mse_loss = torch.nn.MSELoss(reduction="none")
        self.add_module("mse_loss", self.mse_loss)
        self._weight_by = weight_by

    def forward(self, inputs, target, weights=None):
        if weights is None:
            if self._weight_by is None:
                raise Exception(
                    "Cannot call WeightedByMSELoss with no weights without setting weight_by."
                )
            weights = torch.tensor(
                np.array([
                    self._weight_by(actual, prediction)
                    for actual, prediction in zip(target, inputs)
                ]),
                dtype=torch.float32
            )
            # Normalize the weights over the entire batch so they sum to the
            # total number of elements
            num_elements = np.prod(weights.shape)
            weights = num_elements * (weights / weights.sum())

        # Ensure that input, target, and weights have the same shape
        assert inputs.shape == target.shape == weights.shape, (
            "Shapes of input, target, and weights must be the same"
        )

        # Compute element-wise MSE loss (without reducing)
        mse_loss = self.mse_loss(inputs, target)

        weights = weights.to(mse_loss.device)
        # Apply the weights
        weighted_mse_loss = mse_loss * weights

        # Mean of the weighted MSE loss
        loss = torch.mean(weighted_mse_loss)

        return loss


def as_weight_matrix(tensor):
    num_elements = np.prod(tensor.shape)
    return num_elements * (tensor / tensor.sum())


def difference_loss(y_true, y_pred):
    # Compute all pairwise differences - ground truth and predictions
    y_true_diffs = y_true.unsqueeze(2) - y_true.unsqueeze(1)
    y_pred_diffs = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)

    # Calculate the difference loss as the Mean Absolute Error (MAE) between
    # actual and predicted differences
    diff_loss = torch.mean(torch.abs(y_pred_diffs - y_true_diffs), dim=2)

    return diff_loss


def difference_weighted_mse():
    mse = torch.nn.MSELoss(reduction="none")

    def loss(y_pred, y_train):
        weights = as_weight_matrix(difference_loss(y_pred, y_train))
        losses = weights * mse(y_pred, y_train)
        return losses.mean()

    return loss


def normalized_loss(left, right):
    def loss(y_true, y_pred):
        return left(y_true, y_pred) * right(y_true, y_pred)
    return loss


def create_weight_function(
        target_distance, target_weight, min_weight=1.0, max_weight=20.0, use_tau=False
):
    scaling_factor = 2 * float(target_weight) / target_distance

    # Define the function to calculate weights
    def calculate_weights(labels, prediction):
        # Calculate the mean absolute deviation of the labels
        labels = list(labels.cpu())
        prediction = list(prediction.cpu())
        mean = np.mean(labels, axis=0)
        diff = np.abs(labels - mean)

        if use_tau:
            # Convert the regression values into rank-based permutations
            labels_rank = stats.rankdata(labels)
            predictions_rank = stats.rankdata(prediction)

            # Calculate the tau distance between the labels and predictions
            tau, _ = stats.kendalltau(labels_rank, predictions_rank)

            # Scale the tau distance to be in the range [0, 1]
            tau = 0.5 * (tau + 1)
        else:
            tau = 0

        tau_weight = (1 + tau)

        # Incorporate the tau distance into the weight calculation
        weights = scaling_factor * diff * tau_weight

        # Clip the weights to the range [min_weight, max_weight]
        weights = np.clip(weights, min_weight, max_weight)

        return weights

    return calculate_weights
