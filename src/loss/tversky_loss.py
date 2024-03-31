import argparse
import torch
import torch.nn as nn


class TverskyLoss(nn.Module):
    """
    Implements the Tversky Loss function, which is a generalization of the Dice Loss. It adds flexibility by allowing control over the relative importance of false positives and false negatives, making it particularly useful for imbalanced datasets in segmentation tasks.

    The Tversky index (TI) is calculated as:
    TI = TP / (TP + alpha * FP + beta * FN)
    where TP, FP, and FN denote the numbers of true positive, false positive, and false negative pixels, respectively. Alpha and beta control the magnitude of penalties for false positives and false negatives, respectively.

    Attributes:
    | Attribute | Type  | Description                                        |
    |-----------|-------|----------------------------------------------------|
    | smooth    | float | A small constant added to avoid division by zero. |

    Methods:
    - `forward(predicted, actual)`: Computes the Tversky Loss between the predicted and actual segmentation maps.

    Example usage:
    ```python
    tversky_loss = TverskyLoss(smooth=1e-6)
    predicted = torch.randn(10, 1, 256, 256)
    actual = torch.randn(10, 1, 256, 256)
    loss = tversky_loss(predicted, actual)
    print(loss)
    ```
    """

    def __init__(self, smooth=1e-6):
        """
        Initializes the TverskyLoss class with a smoothing factor.

        | Parameter | Type  | Description                                       |
        |-----------|-------|---------------------------------------------------|
        | smooth    | float | Smoothing factor to prevent division by zero. Default: 1e-6. |
        """
        super(TverskyLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predicted, actual):
        """
        Computes the Tversky Loss between the predicted and actual segmentation maps.

        | Parameter  | Type         | Description                        |
        |------------|--------------|------------------------------------|
        | predicted  | torch.Tensor | The predicted segmentation map.    |
        | actual     | torch.Tensor | The ground truth segmentation map. |

        | Returns    | Type         | Description                        |
        |------------|--------------|------------------------------------|
        | torch.Tensor | torch.Tensor | The calculated Tversky loss.      |
        """
        predicted = predicted.contiguous().view(-1)
        actual = actual.contiguous().view(-1)

        tp = (predicted * actual).sum()
        fp = (predicted * (1 - actual)).sum()
        fn = ((1 - predicted) * actual).sum()

        tversky_index = (tp + self.smooth) / (tp + 0.5 * fp + 0.5 * fn + self.smooth)

        return 1 - tversky_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tversky Loss Example")
    parser.add_argument(
        "--smooth",
        type=float,
        default=1e-6,
        help="Smoothing parameter to avoid division by zero in the loss calculation.",
    )
    args = parser.parse_args()

    tversky_loss = TverskyLoss(smooth=args.smooth)

    predicted = torch.tensor([0.8, 0.6, 0.4, 0.2], dtype=torch.float32)
    actual = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    loss = tversky_loss(predicted, actual)

    print(f"Tversky Loss: {loss.item():.4f}")
