import argparse
import torch
import torch.nn as nn


class ComboLoss(nn.Module):
    """
    Implements a combination loss function that includes Binary Cross-Entropy (BCE) Loss, Focal Loss, and Dice Loss. This combined approach is designed to address the issue of class imbalance by focusing training on hard examples and ensuring segmentation tasks benefit from both probabilistic and spatial overlap measures.

    Attributes:
    | Attribute | Type  | Description                                                         |
    |-----------|-------|---------------------------------------------------------------------|
    | smooth    | float | Smoothing constant added to avoid division by zero in Dice Loss.   |
    | alpha     | float | Weighting factor for Focal Loss to balance positive/negative cases.|
    | gamma     | float | Modulating factor for Focal Loss focusing more on hard examples.   |

    Methods:
    - `forward(predicted, actual)`: Computes the combined loss for the predicted and actual segmentation maps.

    Example usage:
    ```python
    combo_loss = ComboLoss(smooth=1e-6, alpha=0.5, gamma=0.5)
    predicted = torch.sigmoid(torch.randn(10, 1, requires_grad=True))
    actual = torch.empty(10, 1).random_(2)
    loss = combo_loss(predicted, actual)
    print(loss)
    ```
    """

    def __init__(self, smooth=1e-6, alpha=0.5, gamma=0.5):
        """
        Initializes the ComboLoss class with smooth, alpha, and gamma parameters.

        | Parameter | Type  | Description                                                       |
        |-----------|-------|-------------------------------------------------------------------|
        | smooth    | float | Smoothing factor for Dice Loss calculation. Default: 1e-6.        |
        | alpha     | float | Alpha parameter for Focal Loss. Default: 0.5.                     |
        | gamma     | float | Gamma parameter for Focal Loss to concentrate on hard examples. Default: 0.5. |
        """
        super(ComboLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predicted, actual):
        """
        Computes the combination loss including BCE Loss, Focal Loss, and Dice Loss between the predicted probabilities and actual targets.

        | Parameter  | Type         | Description                           |
        |------------|--------------|---------------------------------------|
        | predicted  | torch.Tensor | The predicted segmentation map.       |
        | actual     | torch.Tensor | The ground truth segmentation map.    |

        | Returns    | Type         | Description                           |
        |------------|--------------|---------------------------------------|
        | torch.Tensor | torch.Tensor | The calculated combined loss value.   |
        """
        predicted = predicted.view(-1)
        actual = actual.view(-1)

        criterion = nn.BCELoss()
        bce_loss = criterion(predicted, actual)

        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        intersection = (predicted * actual).sum()
        dice_loss = 1 - (2.0 * intersection + self.smooth) / (
            predicted.sum() + actual.sum() + self.smooth
        )

        return bce_loss + focal_loss + dice_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combo Loss Example")
    parser.add_argument(
        "--smooth",
        type=float,
        default=1e-6,
        help="Smoothing constant for Dice Loss",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Alpha value for Focal Loss",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="Gamma value for Focal Loss",
    )

    args = parser.parse_args()

    combo_loss = ComboLoss(smooth=args.smooth, alpha=args.alpha, gamma=args.gamma)

    predicted = torch.tensor([0.8, 0.6, 0.7, 0.9, 0.4], dtype=torch.float32)
    actual = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float32)

    print(f"Total loss using Combo Loss: {combo_loss(predicted, actual).item():.4f}")
