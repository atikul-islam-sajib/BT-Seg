import sys
import os
import logging
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.utils import save_image

sys.path.append("src/")

from config import (
    PROCESSED_PATH,
    TRAIN_CHECKPOINT_PATH,
    TEST_CHECKPOINT_PATH,
    TRAIN_IMAGES_PATH,
    METRICS_PATH,
)
from utils import load, dump, weight_init, define_device
from UNet import UNet


class Trainer:
    """
    A class responsible for training a U-Net model for image segmentation tasks.

    Attributes:
        epochs (int): Number of epochs to train the model.
        lr (float): Learning rate for the optimizer.
        loss (str): Name of the loss function to be used. Currently supports 'dice' loss.
        smooth_value (float): Smoothing value for the Dice loss function to avoid division by zero.
        beta1 (float): The exponential decay rate for the first moment estimates in optimizer.
        device (str): Device on which to train the model (e.g., 'cpu', 'cuda', 'mps').
        is_display (bool): Flag to display progress during training.
        history (dict): A dictionary to record training and testing loss over epochs.

    Methods:
        __setup__(): Sets up the model, loss function, optimizer, and data loaders.
        select_loss_function(): Selects the loss function based on the 'loss' attribute.
        l1_loss(model, lambda_value=0.01): Calculates the L1 loss for regularization.
        l2_loss(model, lambda_value=0.01): Calculates the L2 loss for regularization.
        update_training_loss(**kwargs): Performs a single training step.
        update_testing_loss(**kwargs): Evaluates the model on the test set.
        saved_checkpoints(**kwargs): Saves model checkpoints during training.
        show_progress(**kwargs): Prints or logs the training progress.
        train(): Main method to run the training loop.
        plot_loss_curves(): Static method to plot the training and testing loss curves.

    Examples:
        >>> trainer = Trainer(epochs=10, lr=0.001, loss='dice', device='cuda')
        >>> trainer.train()
    """

    def __init__(
        self,
        epochs=10,
        lr=0.0002,
        loss=None,
        alpha=0.25,
        gamma=2,
        smooth_value=0.01,
        beta1=0.5,
        device="mps",
        display=True,
    ):
        self.epochs = epochs
        self.lr = lr
        self.loss = loss
        self.alpha = alpha
        self.gamma = gamma
        self.smooth_value = smooth_value
        self.beta1 = beta1
        self.beta2 = 0.999
        self.device = device
        self.is_display = display
        self.history = {"train_loss": list(), "test_loss": list()}

    def __setup__(self):
        self.device = define_device(device=self.device)
        self.model = UNet().to(self.device)
        self.model.apply(weight_init)
        self.loss = self.select_loss_function()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        if os.path.exists(PROCESSED_PATH):
            self.train_dataloader = load(
                filename=os.path.join(PROCESSED_PATH, "train_dataloader.pkl")
            )
            self.test_dataloader = load(
                filename=os.path.join(PROCESSED_PATH, "test_dataloader.pkl")
            )
        else:
            raise ValueError(
                "Processed data not found. Please run the preprocessing notebook.".capitalize()
            )

    def select_loss_function(self):
        """
        Selects and returns the appropriate loss function based on the specified loss type during the initialization of the class. This method enables flexible switching between different loss functions for training, depending on the model's requirements and the specific task at hand.

        The method supports various loss types, including 'dice', 'IoU' (Intersection over Union), 'focal', 'dice_bce' (a combination of Dice and Binary Cross-Entropy Loss), 'jaccard', and falls back to standard Binary Cross-Entropy Loss if none of these are specified.

        Returns:
            nn.Module: The selected loss function as a PyTorch module. The specific loss function returned depends on the 'loss' attribute defined during the initialization of the class.

        Supported loss types and their corresponding returned loss functions:
        - 'dice': Returns an instance of the `DiceLoss` class.
        - 'IoU': Returns an instance of the `IoU` class.
        - 'focal': Returns an instance of the `FocalLoss` class, with `alpha` and `gamma` parameters.
        - 'dice_bce': Returns an instance of the `DiceBCE` class.
        - 'jaccard': Returns an instance of the `JaccardLoss` class.
        - Otherwise: Returns an instance of the `nn.BCELoss` class with `reduction` set to "mean".

        Examples:
        ```python
        # Assuming 'self.loss' is set to "dice" and 'self.smooth_value' is defined
        loss_function = self.select_loss_function()
        # `loss_function` is now an instance of `DiceLoss`

        # If 'self.loss' is set to "focal", with 'self.alpha' and 'self.gamma' defined
        loss_function = self.select_loss_function()
        # `loss_function` is now an instance of `FocalLoss`
        ```
        """
        if self.loss == "dice":
            return DiceLoss(smooth=self.smooth_value)
        elif self.loss == "IoU":
            return IoU(smooth=self.smooth_value)
        elif self.loss == "focal":
            return FocalLoss(alpha=self.alpha, gamma=self.gamma)
        elif self.loss == "dice_bce":
            return DiceBCE(smooth=self.smooth_value)
        elif self.loss == "jaccard":
            return JaccardLoss(smooth=self.smooth_value)
        elif self.loss == "combo":
            return ComboLoss(
                alpha=self.alpha, gamma=self.gamma, smooth=self.smooth_value
            )
        else:
            print(self.loss)
            return nn.BCELoss(reduction="mean")

    def l1_loss(self, model, lambda_value=0.01):
        """
        Calculates the L1 regularization loss.

        Parameters:
            model (torch.nn.Module): The model whose weights are regularized.
            lambda_value (float): Regularization coefficient.

        Returns:
            torch.Tensor: The L1 regularization loss.
        """
        return lambda_value * sum(
            (torch.norm(input=params, p=1) for params in model.parameters())
        )

    def l2_loss(self, model, lambda_value=0.01):
        """
        Calculates the L2 regularization loss.

        Parameters:
            model (torch.nn.Module): The model whose weights are regularized.
            lambda_value (float): Regularization coefficient.

        Returns:
            torch.Tensor: The L2 regularization loss.
        """
        return lambda_value * sum(
            (torch.norm(input=params, p=2) for params in model.parameters())
        )

    def update_training_loss(self, **kwargs):
        """
        Updates the model's weights by performing a single step of training.

        Parameters:
            kwargs (dict): Contains 'images' and 'masks', both of type torch.Tensor, representing
                           the input images and their corresponding ground truth masks.

        Returns:
            float: The training loss for the current step.
        """
        self.optimizer.zero_grad()

        train_predicted_masks = self.model(kwargs["images"])
        train_predicted_loss = self.loss(train_predicted_masks, kwargs["masks"])

        train_predicted_loss.backward(retain_graph=True)
        self.optimizer.step()

        return train_predicted_loss.item()

    def update_testing_loss(self, **kwargs):
        """
        Computes the loss on the test dataset without updating the model's weights.

        Parameters:
            kwargs (dict): Contains 'images' and 'masks', both of type torch.Tensor, representing
                           the input images and their corresponding ground truth masks.

        Returns:
            float: The testing loss for the current step.
        """
        test_predicted_masks = self.model(kwargs["images"])
        test_predicted_loss = self.loss(test_predicted_masks, kwargs["masks"])

        return test_predicted_loss.item()

    def saved_checkpoints(self, **kwargs):
        """
        Saves the model's checkpoints at specified intervals.

        Parameters:
            kwargs (dict): Contains 'epoch', the current epoch during training.
        """
        if kwargs["epoch"] != self.epochs:
            if os.path.exists(TRAIN_CHECKPOINT_PATH):
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        TRAIN_CHECKPOINT_PATH,
                        "model_{}.pth".format(kwargs["epoch"]),
                    ),
                )
        else:
            if os.path.exists(TEST_CHECKPOINT_PATH):
                torch.save(
                    self.model.state_dict(),
                    os.path.join(TEST_CHECKPOINT_PATH, "best_model.pth"),
                )

    def show_progress(self, **kwargs):
        """
        Displays or logs the training progress, including the current epoch and loss.

        Parameters:
            kwargs (dict): Contains 'epoch', 'epochs', 'train_loss', and 'test_loss', detailing
                           the current epoch, total epochs, training loss, and testing loss, respectively.
        """
        if self.is_display == True:
            print(
                "Epochs: [{}/{}] - train_loss: [{:.5f}] - test_loss: [{:.5f}]".format(
                    kwargs["epoch"],
                    kwargs["epochs"],
                    kwargs["train_loss"],
                    kwargs["test_loss"],
                )
            )
        elif self.is_display == False:
            logging.info(
                "Epochs: [{}/{}] - train_loss: [{:.5f}] - test_loss: [{:.5f}]".format(
                    kwargs["epoch"],
                    kwargs["epochs"],
                    kwargs["train_loss"],
                    kwargs["test_loss"],
                )
            )

    def train(self):
        """
        Executes the training loop for the model over a specified number of epochs. This method performs several key functions:

        - Calls the setup method to prepare the model, optimizer, and data loaders.
        - Iterates over the training and testing datasets for each epoch to compute loss and update model parameters.
        - Saves model checkpoints at each epoch and the best model at the end of training.
        - Updates the training history with loss metrics for visualization.
        - Optionally saves images of predicted masks for visual inspection.

        The training process can be interrupted by exceptions related to file paths or IO operations. These are caught and reported to allow for troubleshooting.

        Raises:
            Exception: If the setup process (`__setup__`) fails, indicating issues with data loading or model initialization.
            Exception: If saving checkpoints or metrics fails due to issues with the file path or writing permissions.
            Exception: If there's an issue saving the training progress images, indicating a problem with the specified path.

        Note:
            This method updates the model's weights and records the training and testing loss in the `history` attribute.
            It also saves checkpoints and training images to the filesystem for later analysis or resumption of training.

        Examples:
            ```python
            trainer = Trainer(...)
            trainer.train()  # Starts the training process
            ```
        """
        try:
            self.__setup__()
        except Exception as e:
            print("The exception in the section # {}".format(e).capitalize())
        else:
            self.model.train()
            for epoch in tqdm(range(self.epochs)):
                train_loss = list()
                test_loss = list()

                for _, (images, masks) in enumerate(self.train_dataloader):
                    images = images.to(self.device)
                    masks = masks.to(self.device)

                    train_loss.append(
                        self.update_training_loss(images=images, masks=masks)
                    )

                self.model.eval()

                for _, (images, masks) in enumerate(self.test_dataloader):
                    images = images.to(self.device)
                    masks = masks.to(self.device)

                    test_loss.append(
                        self.update_testing_loss(images=images, masks=masks)
                    )

                try:
                    self.saved_checkpoints(epoch=epoch + 1)

                    self.history["train_loss"].append(np.mean(train_loss))
                    self.history["test_loss"].append(np.mean(test_loss))

                except Exception as e:
                    print(e)
                else:
                    images, _ = next(iter(self.test_dataloader))
                    predicted_masks = self.model(images.to(self.device))
                    if os.path.exists(TRAIN_IMAGES_PATH):
                        save_image(
                            predicted_masks,
                            os.path.join(
                                TRAIN_IMAGES_PATH,
                                "train_masks_{}.png".format(epoch + 1),
                            ),
                            nrow=6,
                            normalize=True,
                        )
                    else:
                        raise Exception("Train images path not found.".capitalize())
                finally:
                    self.show_progress(
                        epoch=epoch + 1,
                        epochs=self.epochs,
                        train_loss=np.mean(train_loss),
                        test_loss=np.mean(test_loss),
                    )

            if os.path.exists(METRICS_PATH):
                dump(
                    value=self.history,
                    filename=os.path.join(METRICS_PATH, "metrics.pkl"),
                )
            else:
                raise Exception("Metrics path not found.".capitalize())

    @staticmethod
    def plot_loss_curves():
        """
        Plots the training and testing loss curves. Requires the metrics to be saved in a specified path.

        Raises:
            Exception: If the metrics path is not found.
        """
        if os.path.exists(METRICS_PATH):
            history = load(filename=os.path.join(METRICS_PATH, "metrics.pkl"))
            plt.figure(figsize=(15, 10))

            plt.plot(history["train_loss"], label="Train Loss".title())
            plt.plot(history["test_loss"], label="Test Loss".title())

            plt.title("Loss Curves".title())
            plt.xlabel("Epochs".title())
            plt.ylabel("Loss".title())
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            raise Exception("Metrics path not found.".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trainer".title())
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs".capitalize()
    )
    parser.add_argument(
        "--lr", type=float, default=1e-2, help="Learning rate".capitalize()
    )
    parser.add_argument(
        "--loss", type=str, default="dice", help="Loss function".capitalize()
    )
    parser.add_argument(
        "--display", type=bool, default=True, help="Display progress".capitalize()
    )
    parser.add_argument("--device", type=str, default="mps", help="Device".capitalize())
    parser.add_argument(
        "--smooth_value", type=float, default=0.01, help="Smooth value".capitalize()
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Alpha value".capitalize()
    )
    parser.add_argument(
        "--gamma", type=float, default=2, help="Gamma value".capitalize()
    )
    parser.add_argument("--train", action="store_true", help="Train model".capitalize())

    args = parser.parse_args()

    if args.train:
        if (
            args.epochs
            and args.lr
            and args.loss
            and args.display
            and args.device
            and args.smooth_value
            and args.alpha
            and args.gamma
        ):
            trainer = Trainer(
                epochs=args.epochs,
                lr=args.lr,
                loss=args.loss,
                alpha=args.alpha,
                gamma=args.gamma,
                display=args.display,
                device=args.device,
                smooth_value=args.smooth_value,
            )
            trainer.train()
    else:
        raise Exception("Train flag is not set.".capitalize())
