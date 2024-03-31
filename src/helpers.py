import os
import torch
import torch.nn as nn
import torch.optim as optim

from utils import load
from config import PROCESSED_PATH
from UNet import UNet
from AttentionUNet import AttentionUNet

from loss.dice_loss import DiceLoss
from loss.jaccard_loss import JaccardLoss
from loss.focal_loss import FocalLoss
from loss.combo_loss import ComboLoss


def helpers(**kwargs):
    """
    Initializes and returns the model, optimizer, loss function, and dataloaders for training and testing.

    This function sets up the model (either AttentionUNet or UNet), optimizer, and criterion based on the passed
    arguments. It also loads the train and test dataloaders if the processed data path exists.

    Parameters:
    - **kwargs: Keyword arguments containing the setup configuration. Expected keys are:
        - is_attentionUNet (bool): If True, uses the AttentionUNet model, otherwise UNet.
        - device (torch.device): The device to run the model on (e.g., 'cpu', 'cuda').
        - loss (str): The name of the loss function to use. Options include 'dice', 'dice_bce', 'IoU' (Jaccard loss),
          'focal', and 'combo'. Anything else defaults to binary cross-entropy loss.
        - lr (float): Learning rate for the optimizer.
        - smooth (float, optional): Smoothing parameter for Dice and Jaccard loss. Default value depends on the loss function implementation.
        - gamma (float, optional): Gamma parameter for Focal loss and Combo loss. Default value depends on the loss function implementation.
        - alpha (float, optional): Alpha parameter for Focal loss and Combo loss. Default value depends on the loss function implementation.

    Raises:
    - Exception: If the processed data path does not exist, preventing the loading of train and test dataloaders.

    Returns:
    - dict: A dictionary containing the initialized model, optimizer, criterion, train dataloader, and test dataloader.
    """
    is_attentionUNet = kwargs["is_attentionUNet"]
    device = device = kwargs["device"]
    loss = kwargs["loss"]

    if is_attentionUNet == True:
        model = AttentionUNet().to(device)
    else:
        model = UNet().to(device)

    optimizer = optim.Adam(model.parameters(), lr=kwargs["lr"], betas=(0.5, 0.999))

    if os.path.exists(PROCESSED_PATH):
        train_dataloader = load(
            filename=os.path.join(PROCESSED_PATH, "train_dataloader.pkl")
        )
        test_dataloader = load(
            filename=os.path.join(PROCESSED_PATH, "test_dataloader.pkl")
        )
    else:
        raise Exception("Dataloader - train & test cannot be loaded".capitalize())

    if loss == "dice":
        criterion = DiceLoss(smooth=kwargs["smooth"])
    elif loss == "dice_bce":
        criterion = DiceLoss(smooth=kwargs["smooth"])
    elif loss == "IoU":
        criterion = JaccardLoss(smooth=kwargs["smooth"])
    elif loss == "focal":
        criterion = FocalLoss(gamma=kwargs["gamma"], alpha=kwargs["alpha"])
    elif loss == "combo":
        criterion = ComboLoss(
            smooth=kwargs["smooth"], alpha=kwargs["alpha"], gamma=kwargs["gamma"]
        )
    else:
        criterion = nn.BCELoss()

    return {
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "train_dataloader": train_dataloader,
        "test_dataloader": test_dataloader,
    }
