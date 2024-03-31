import argparse
import torch
import torch.nn as nn


class JaccardLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predicted, actual):
        predicted = predicted.view(-1)
        actual = actual.view(-1)

        intersection = (predicted * actual).sum()

        return 1 - (2 * intersection + self.smooth) / (
            predicted.sum() + actual.sum() - intersection + self.smooth
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jaccard Loss Example".title())
    parser.add_argument(
        "--smooth", type=float, default=1e-6, help="Smoothing factor (default: 1e-6)"
    )
    args = parser.parse_args()

    if args.smooth:
        IoU = JaccardLoss(smooth=args.smooth)

        predicted = torch.tensor([0.2, 0.3, 0.5])
        actual = torch.tensor([0.2, 0.3, 0.5])

        loss = IoU(predicted, actual)

        print("The loss of the IoU {}".format(loss))

    else:
        raise Exception("Smoothing factor must be provided.".capitalize())
