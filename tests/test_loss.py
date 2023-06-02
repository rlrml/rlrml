import torch

from rlrml import loss


def test_combined_loss_masking():
    combined = loss.difference_and_mse_loss()

    # Initialize tensors and mask
    y_true = torch.tensor([[1.0, 2.0, 3.0, 4.5, 1.0]])
    y_pred = torch.tensor([[1.5, 2.5, 3.5, 5.0, 6.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0]])

    assert torch.allclose(
        combined(y_true, y_pred, mask) * mask,
        torch.nn.MSELoss(reduction='none')(y_true, y_pred) * mask
    )


def test_difference_loss_only_cares_about_differences():
    diff_loss = loss.DifferenceLoss()

    # Initialize tensors and mask
    y_true = torch.tensor([[1.0, 2.0, 3.0, 4.5]])
    y_pred = torch.tensor([[1.5, 2.5, 3.5, 5.0]])

    loss_result = diff_loss(y_true, y_pred)

    assert torch.allclose(loss_result, torch.FloatTensor([[0.0, 0.0, 0.0, 0.0]]))


def test_difference_loss_masking():
    diff_loss = loss.DifferenceLoss()

    # Initialize tensors and mask
    y_true = torch.tensor([[1.0, 2.0, 3.0, 4.5], [4.0, 5.0, 6.0, 7.5]])
    y_pred = torch.tensor([[1.5, 2.5, 3.5, 5.0], [4.5, 5.5, 6.5, 7.5]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0]])

    loss_result = diff_loss(y_true, y_pred, mask)

    # Change values where mask is 0, result should be the same
    y_true_masked = y_true.clone()
    y_true_masked[mask == 0] = 999.0

    y_pred_masked = y_pred.clone()
    y_pred_masked[mask == 0] = -999.0

    diff_loss_masked = diff_loss(y_true_masked, y_pred_masked, mask)

    assert torch.allclose(loss_result, diff_loss_masked)

    # The changes do actually make a difference with no mask
    diff_loss_unmasked = diff_loss(y_true_masked, y_pred_masked)

    assert not torch.allclose(diff_loss_masked, diff_loss_unmasked)
