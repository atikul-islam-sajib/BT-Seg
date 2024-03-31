import argparse
import torch
import torch.nn as nn


class ComboLoss(nn.Module):
    def __init__(self, smooth=1e-6, alpha=0.5, gamma=0.5):
        super(ComboLoss, self).__init__()

        self.smooth = smooth
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predicted, actual):
        predicted = predicted.view(-1)
        actual = actual.view(-1)

        criterion = nn.BCELoss()
        BCELoss = criterion(predicted, actual)

        pt = torch.exp(-BCELoss)
        FocalLoss = self.alpha * (1 - pt) ** self.gamma * BCELoss

        intersection = (predicted * actual).sum()

        DiceLoss = 1 - (2.0 * intersection + self.smooth) / (
            predicted.sum() + actual.sum() + self.smooth
        )

        return (BCELoss + FocalLoss) + DiceLoss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combo Loss Example".title())
    parser.add_argument(
        "--smooth",
        type=float,
        default=1e-6,
        help="Smoothing constant for Dice Loss",
        required=True,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Alpha value for Focal Loss",
        required=True,
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="Gamma value for Focal Loss",
        required=True,
    )

    args = parser.parse_args()

    if args.smooth and args.alpha and args.gamma:

        loss = ComboLoss(args.smooth, args.alpha, args.gamma)

        predicted = torch.tensor([0.8, 0.6, 0.7, 0.9, 0.4])
        actual = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0])

        print(loss(predicted, actual))

    else:
        raise Exception("Arguments should be provided".capitalize())
