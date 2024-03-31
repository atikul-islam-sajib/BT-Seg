import argparse
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    Implements the Focal Loss function, designed to address class imbalance by down-weighting well-classified examples and focusing on hard, misclassified examples.

    Attributes:
    | Attribute | Type  | Description                                                         |
    |-----------|-------|---------------------------------------------------------------------|
    | alpha     | float | A balancing factor for the negative and positive classes.          |
    | gamma     | float | A focusing parameter to adjust the rate at which easy examples are down-weighted. |

    Methods:
    - `forward(predicted, actual)`: Computes the Focal Loss between the predicted probabilities and actual binary labels.

    Example usage:
    ```python
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    predicted = torch.sigmoid(torch.randn(10, 1, requires_grad=True))
    actual = torch.empty(10, 1).random_(2)
    loss = focal_loss(predicted, actual)
    loss.backward()
    print(loss)
    ```
    """

    def __init__(self, alpha=0.25, gamma=2):
        """
        Initializes the FocalLoss class with the alpha and gamma parameters.

        | Parameter | Type  | Description                                                |
        |-----------|-------|------------------------------------------------------------|
        | alpha     | float | Weighting factor for the class labels. Default: 0.25.      |
        | gamma     | float | Modulating factor to dynamically scale the loss. Default: 2. |
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predicted, actual):
        """
        Computes the Focal Loss between the predicted probabilities and the actual targets.

        | Parameter  | Type         | Description                             |
        |------------|--------------|-----------------------------------------|
        | predicted  | torch.Tensor | The predicted probabilities.            |
        | actual     | torch.Tensor | The actual targets.                     |

        | Returns    | Type         | Description                             |
        |------------|--------------|-----------------------------------------|
        | torch.Tensor | torch.Tensor | The calculated Focal Loss.              |
        """
        predicted = predicted.view(-1)
        actual = actual.view(-1)

        criterion = nn.BCELoss(reduction="none")
        BCE_loss = criterion(predicted, actual)
        pt = torch.exp(-BCE_loss)

        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        return focal_loss.mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Focal Loss Example")
    parser.add_argument(
        "--alpha", type=float, default=0.25, help="Alpha parameter for Focal Loss"
    )
    parser.add_argument(
        "--gamma", type=float, default=2, help="Gamma parameter for Focal Loss"
    )

    args = parser.parse_args()

    focal_loss = FocalLoss(alpha=args.alpha, gamma=args.gamma)

    predicted = torch.tensor([0.8, 0.6, 0.2], dtype=torch.float32)
    actual = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

    loss = focal_loss(predicted, actual)

    print(f"Total loss using Focal Loss: {loss.item():.4f}")
