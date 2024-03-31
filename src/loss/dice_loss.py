import sys
import os
import argparse
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Implements the Dice Loss for evaluating segmentation performance, particularly useful for binary segmentation tasks.

    Dice Loss is a measure of overlap between two samples. It ranges from 0 to 1 where a Dice Loss of 1 denotes perfect and complete overlap.

    Attributes:
    | Attribute | Type  | Description                            |
    |-----------|-------|----------------------------------------|
    | smooth    | float | A smoothing factor to avoid division by zero. |

    Methods:
    - `forward(predicted, actual)`: Computes the Dice Loss between predicted and actual segmentations.

    Example usage:
    ```python
    dice_loss = DiceLoss(smooth=1e-6)
    predicted = torch.randn(10, 1, 256, 256)
    actual = torch.randn(10, 1, 256, 256)
    loss = dice_loss(predicted, actual)
    print(loss)
    ```
    """

    def __init__(self, smooth=1e-6):
        """
        Initializes the DiceLoss class with a smoothing factor.

        | Parameter | Type  | Description                              |
        |-----------|-------|------------------------------------------|
        | smooth    | float | Smoothing factor to prevent division by zero. Default: 1e-6. |
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predicted, actual):
        """
        Computes the Dice Loss between predicted and actual segmentations.

        | Parameter  | Type          | Description                     |
        |------------|---------------|---------------------------------|
        | predicted  | torch.Tensor  | The predicted segmentation map. |
        | actual     | torch.Tensor  | The ground truth segmentation map. |

        | Returns    | Type          | Description                     |
        |------------|---------------|---------------------------------|
        | dice       | torch.Tensor  | The calculated Dice Loss.       |
        """
        predicted = predicted.contiguous().view(-1)
        actual = actual.contiguous().view(-1)

        intersection = (predicted * actual).sum()
        dice = (2.0 * intersection + self.smooth) / (
            predicted.sum() + actual.sum() + self.smooth
        )

        return (
            1 - dice
        )  # Typically, 1 - Dice coefficient is returned as the loss to minimize.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dice Loss Test")
    parser.add_argument(
        "--smooth",
        type=float,
        default=1e-6,
        help="Smooth value to avoid division by zero in Dice Loss calculation.",
        required=False,  # Making smooth parameter not required for simplicity.
    )
    args = parser.parse_args()

    dice_loss = DiceLoss(smooth=args.smooth)

    predicted = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    actual = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])

    print(dice_loss(predicted, actual))
