import sys
import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn


class DiceBCE(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceBCE, self).__init__()

        self.smooth = smooth

    def forward(self, predicted, actual):
        predicted = predicted.contiguous().view(-1)
        actual = actual.contiguous().view(-1)

        intersection = (predicted * actual).sum()
        dice = (2.0 * intersection + self.smooth) / (
            predicted.sum() + actual.sum() + self.smooth
        )

        criterion = nn.BCELoss()

        return dice + criterion(predicted, actual)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiceBCE Loss")
    parser.add_argument(
        "--smooth", type=float, default=1e-6, help="Smoothing factor for dice loss"
    )

    args = parser.parse_args()

    if args.smooth:
        dice_bce = DiceBCE(smooth=args.smooth)

        predicted = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float)
        actual = torch.tensor([0, 1, 1], dtype=torch.float)

        print(dice_bce(predicted, actual))

    else:
        raise Exception("Arguments should be provided".capitalize())
