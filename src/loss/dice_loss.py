import sys
import os
import argparse
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predicted, actual):
        predicted = predicted.contiguous().view(-1)
        actual = actual.contiguous().view(-1)

        intersection = (predicted * actual).sum()
        dice = (2.0 * intersection + self.smooth) / (
            predicted.sum() + actual.sum() + self.smooth
        )

        return dice


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dice Loss Test".capitalize())
    parser.add_argument(
        "--smooth",
        type=float,
        default=1e-6,
        help="Define the smooth value".capitalize(),
        required=True,
    )
    args = parser.parse_args()

    if args.smooth:
        dice = DiceLoss(smooth=args.smooth)

        predicted = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        actual = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])

        print(dice(predicted, actual))

    else:
        raise Exception("Arguments should be provided".capitalize())
