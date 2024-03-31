import sys
import os
import argparse
import torch
import torch.nn as nn


class DiceBCE(nn.Module):
    """
    Implements a combination of Dice Loss and Binary Cross-Entropy (BCE) Loss, commonly used in binary segmentation tasks.

    This combined loss function leverages the advantages of both Dice Loss, which is good for dealing with class imbalance, and BCE Loss, which calculates the cross-entropy loss between the predicted and actual segmentation maps.

    Attributes:
    | Attribute | Type  | Description                                             |
    |-----------|-------|---------------------------------------------------------|
    | smooth    | float | A smoothing factor to avoid division by zero in Dice calculation. |

    Methods:
    - `forward(predicted, actual)`: Computes the combined Dice and BCE loss.

    Example usage:
    ```python
    model = SomeModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = DiceBCE(smooth=1e-6)

    # Assuming `images` and `masks` are your inputs and targets, respectively
    outputs = model(images)
    loss = criterion(outputs, masks)
    loss.backward()
    optimizer.step()
    ```

    """

    def __init__(self, smooth=1e-6):
        """
        Initializes the DiceBCE class with a smoothing factor for the Dice loss component.

        | Parameter | Type  | Description                                        |
        |-----------|-------|----------------------------------------------------|
        | smooth    | float | Smoothing factor for Dice calculation. Default: 1e-6. |
        """
        super(DiceBCE, self).__init__()
        self.smooth = smooth

    def forward(self, predicted, actual):
        """
        Computes the combined Dice and Binary Cross-Entropy (BCE) Loss.

        | Parameter  | Type         | Description                           |
        |------------|--------------|---------------------------------------|
        | predicted  | torch.Tensor | The predicted segmentation map.       |
        | actual     | torch.Tensor | The ground truth segmentation map.    |

        | Returns    | Type         | Description                           |
        |------------|--------------|---------------------------------------|
        | torch.Tensor | torch.Tensor | The calculated combined Dice and BCE loss. |
        """
        predicted = predicted.contiguous().view(-1)
        actual = actual.contiguous().view(-1)

        intersection = (predicted * actual).sum()
        dice_loss = (2.0 * intersection + self.smooth) / (
            predicted.sum() + actual.sum() + self.smooth
        )

        bce_loss = nn.BCELoss()(predicted, actual)

        return 1 - dice_loss + bce_loss  # Combining 1 - Dice Loss and BCE Loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiceBCE Loss Test")
    parser.add_argument(
        "--smooth",
        type=float,
        default=1e-6,
        help="Smoothing factor for Dice loss calculation.",
    )

    args = parser.parse_args()

    dice_bce_loss = DiceBCE(smooth=args.smooth)

    predicted = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float)
    actual = torch.tensor([0, 1, 1], dtype=torch.float)

    print(dice_bce_loss(predicted, actual))
