import argparse
import torch
import torch.nn as nn


class TverskyLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(TverskyLoss, self).__init__()

        self.smooth = smooth

    def forward(self, predicted, actual):
        predicted = predicted.contiguous().view(-1)
        actual = actual.contiguous().view(-1)

        tp = (predicted * actual).sum()
        fp = (actual * (1 - predicted)).sum()
        fn = ((1 - actual) * predicted).sum()

        tversky = (tp + self.smooth) / (tp + fp + fn + self.smooth)

        return 1 - tversky


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tversky Loss Example")
    parser.add_argument(
        "--smooth",
        type=float,
        default=1e-6,
        help="smoothing parameter",
    )
    args = parser.parse_args()

    if args.smooth:
        tversky = TverskyLoss(smooth=args.smooth)

        predicted = torch.tensor([0.8, 0.6, 0.4, 0.2])
        actual = torch.tensor([1.0, 0.0, 0.0, 0.0])

        print(tversky(predicted, actual))

    else:
        raise Exception("Please provide a smoothing parameter.")
