import argparse
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predicted, actual):
        predicted = predicted.view(-1)
        actual = actual.view(-1)

        criterion = nn.BCELoss()
        BCE = criterion(predicted, actual)
        pt = torch.exp(-BCE)

        return self.alpha * (1 - pt) ** self.gamma * criterion(predicted, actual)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Focal Loss Example".title())
    parser.add_argument(
        "--alpha", type=float, default=0.25, help="Alpha parameter for Focal Loss"
    )
    parser.add_argument(
        "--gamma", type=float, default=2, help="Gamma parameter for Focal Loss"
    )

    args = parser.parse_args()

    if args.alpha and args.gamma:
        focal_loss = FocalLoss(alpha=args.alpha, gamma=args.gamma)

        predicted = torch.tensor([0.8, 0.6, 0.2])
        actual = torch.tensor([1.0, 0.0, 0.0])

        loss = focal_loss(predicted, actual)

        print("Total loss using focal loss {}".format(loss))

    else:
        raise Exception("Arguments should be provided for alpha and gamma".capitalize())
