import argparse
import torch
import torch.nn as nn


class JaccardLoss(nn.Module):
    """
    Implements the Jaccard Loss, also known as the Intersection over Union (IoU) Loss, for evaluating segmentation models. This loss is particularly useful for tasks where the positive class (foreground) is much smaller than the negative class (background), as it normalizes over the size of both classes.

    The Jaccard Index measures the similarity between two sets, and its loss is a useful metric for the evaluation of image segmentation tasks.

    Attributes:
    | Attribute | Type  | Description                                            |
    |-----------|-------|--------------------------------------------------------|
    | smooth    | float | A small constant added to the numerator and denominator to avoid division by zero. |

    Methods:
    - `forward(predicted, actual)`: Computes the Jaccard Loss between the predicted and actual segmentation maps.

    Example usage:
    ```python
    jaccard_loss = JaccardLoss(smooth=1e-6)
    predicted = torch.randn(10, 1, 256, 256)
    actual = torch.randn(10, 1, 256, 256)
    loss = jaccard_loss(predicted, actual)
    print(loss)
    ```
    """

    def __init__(self, smooth=1e-6):
        """
        Initializes the JaccardLoss class with a smoothing factor.

        | Parameter | Type  | Description                                         |
        |-----------|-------|-----------------------------------------------------|
        | smooth    | float | Smoothing factor to prevent division by zero. Default: 1e-6. |
        """
        super(JaccardLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predicted, actual):
        """
        Computes the Jaccard Loss (IoU Loss) between the predicted and actual segmentation maps.

        | Parameter  | Type         | Description                          |
        |------------|--------------|--------------------------------------|
        | predicted  | torch.Tensor | The predicted segmentation map.      |
        | actual     | torch.Tensor | The ground truth segmentation map.   |

        | Returns    | Type         | Description                          |
        |------------|--------------|--------------------------------------|
        | torch.Tensor | torch.Tensor | The calculated Jaccard (IoU) loss.  |
        """
        predicted = predicted.view(-1)
        actual = actual.view(-1)

        intersection = (predicted * actual).sum()
        total_union = predicted.sum() + actual.sum() - intersection

        return 1 - (intersection + self.smooth) / (total_union + self.smooth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jaccard Loss Example")
    parser.add_argument(
        "--smooth",
        type=float,
        default=1e-6,
        help="Smoothing factor for Jaccard Loss calculation.",
    )

    args = parser.parse_args()

    iou_loss = JaccardLoss(smooth=args.smooth)

    predicted = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float32)
    actual = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float32)

    loss = iou_loss(predicted, actual)

    print(f"The loss of the IoU {loss.item():.4f}")
