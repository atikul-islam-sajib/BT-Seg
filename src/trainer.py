import os
import logging
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.utils import save_image

from config import (
    TRAIN_CHECKPOINT_PATH,
    TEST_CHECKPOINT_PATH,
    TRAIN_IMAGES_PATH,
    METRICS_PATH,
)
from utils import load, dump, device_init
from helpers import helpers


class Trainer:
    """
    A trainer class for training a U-Net or Attention U-Net model for image segmentation tasks.

    | Parameter         | Type      | Description                                                                                                                                               |
    |-------------------|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
    | epochs            | int       | The number of training epochs. Default is 10.                                                                                                             |
    | lr                | float     | The learning rate for the optimizer. Default is 1e-4.                                                                                                     |
    | loss              | str       | The loss function to use. Options include 'dice', 'dice_bce', 'IoU', 'focal', and 'combo'. Defaults to binary cross-entropy if not specified.            |
    | is_attentionUNet  | bool      | If True, uses the AttentionUNet model, otherwise uses U-Net. Default is False.                                                                            |
    | is_l1             | bool      | If True, applies L1 regularization. Default is False.                                                                                                     |
    | is_l2             | bool      | If True, applies L2 regularization. Default is False.
    | is_elastic             | bool      | If True, applies elastic regularization. Default is False. |
    | is_weight_clip    | bool      | If True, applies weight clipping to the model's parameters. Default is False.                                                                             |
    | alpha             | float     | The alpha parameter for Focal loss and Combo loss. Default is 0.25.                                                                                       |
    | gamma             | float     | The gamma parameter for Focal loss and Combo loss. Default is 2.                                                                                          |
    | smooth            | float     | The smoothing parameter for Dice and Jaccard loss. Default is 0.01.                                                                                       |
    | beta1             | float     | The exponential decay rate for the first moment estimates in Adam optimizer. Default is 0.5.                                                              |
    | beta2             | float     | The exponential decay rate for the second moment estimates in Adam optimizer. Default is 0.999.                                                           |
    | min_clip          | float     | The minimum value for weight clipping. Default is -0.01.                                                                                                   |
    | max_clip          | float     | The maximum value for weight clipping. Default is 0.01.                                                                                                    |
    | device            | str       | The device to run the model on (e.g., 'cpu', 'cuda', 'mps'). Default is 'mps'.                                                                             |
    | display           | bool      | If True, displays training progress. Default is True.                                                                                                      |

    ### Notes:
    - The device is automatically initialized based on the given `device` string.
    - Weight clipping is applied if `is_weight_clip` is True, with limits set by `min_clip` and `max_clip`.

    ### Examples:
    ```python
    trainer = Trainer(
        epochs=20,
        lr=0.001,
        loss='dice',
        is_attentionUNet=True,
        device='cuda'
    )
    trainer.train()
    ```

    This example initializes a Trainer object for training an Attention U-Net model with the Dice loss function on a CUDA device.
    """

    def __init__(
        self,
        epochs=50,
        lr=1e-2,
        loss=None,
        is_attentionUNet=False,
        is_l1=False,
        is_l2=False,
        is_elastic=False,
        is_weight_clip=False,
        smooth=0.01,
        alpha=0.25,
        gamma=2,
        beta1=0.9,
        beta2=0.999,
        device="mps",
        display=True,
    ):

        self.epochs = epochs
        self.lr = lr
        self.loss = loss
        self.is_attentionUNet = is_attentionUNet
        self.is_l1 = is_l1
        self.is_l2 = is_l2
        self.is_elastic = is_elastic
        self.is_weight_clip = is_weight_clip
        self.smooth = smooth
        self.alpha = alpha
        self.gamma = gamma
        self.beta1 = beta1
        self.beta2 = beta2
        self.device = device_init(device)
        self.display = display
        self.history = {"train_loss": list(), "test_loss": list()}

        self.setup = helpers(
            is_attentionUNet=self.is_attentionUNet,
            device=self.device,
            lr=self.lr,
            loss=self.loss,
            smooth=self.smooth,
            alpha=self.alpha,
            gamma=self.gamma,
        )

        self.model = self.setup["model"]
        self.optimizer = self.setup["optimizer"]
        self.criterion = self.setup["criterion"]
        self.train_dataloader = self.setup["train_dataloader"]
        self.test_dataloader = self.setup["test_dataloader"]

    def l1(self, model, value=0.01):
        """
        Calculates the L1 regularization loss.

        Parameters:
            model (torch.nn.Module): The model whose weights are regularized.
            lambda_value (float): Regularization coefficient.

        Returns:
            torch.Tensor: The L1 regularization loss.
        """
        if model is not None:
            return value * sum(torch.norm(params, 1) for params in model.parameters())
        else:
            raise Exception("model should be defined".capitalize())

    def l2(self, model, value=0.01):
        """
        Calculates the L2 regularization loss.

        Parameters:
            model (torch.nn.Module): The model whose weights are regularized.
            lambda_value (float): Regularization coefficient.

        Returns:
            torch.Tensor: The L2 regularization loss.
        """
        if model is not None:
            return value * sum(torch.norm(params, 2) for params in model.parameters())
        else:
            raise Exception("model should be defined".capitalize())

    def elastic_net(self, model, value=0.01):
        """
        Calculate the Elastic Net regularization term for a given model.

        Elastic Net regularization is a linear combination of the L1 and L2
        regularization terms. It is used to penalize complex models to prevent
        overfitting. This method computes the Elastic Net regularization term
        as a weighted sum of the L1 and L2 penalties of the model parameters.

        Parameters:
        - model: The model for which to compute the Elastic Net regularization. The model
        should have methods or attributes that allow for the calculation of L1 and L2
        regularization terms.
        - value: A float specifying the weighting of the Elastic Net term. Defaults to 0.01.

        Returns:
        - The Elastic Net regularization term as a float.

        Raises:
        - Exception: If the `model` argument is None, indicating that no model was provided.

        Example:
        - elastic_net_value = instance.elastic_net(model=my_model, value=0.01)
        This would calculate the Elastic Net regularization term for `my_model` with a
        weight of 0.01.
        """
        if model is not None:
            return value * (self.l1(model=model) + self.l2(model=model))
        else:
            raise Exception("model should be defined".capitalize())

    def update_train(self, **kwargs):
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

        train_predicted_loss = self.criterion(train_predicted_masks, kwargs["masks"])

        if self.is_l1 == True:
            train_predicted_loss += self.l1(model=self.model)

        if self.is_l2 == True:
            train_predicted_loss += self.l2(model=self.model)

        if self.is_elastic == True:
            train_predicted_loss += self.elastic_net(model=self.model)

        if self.is_weight_clip == True:
            for params in self.model.parameters():
                params.data.clamp_(-0.10, 0.01)

        train_predicted_loss.backward()

        self.optimizer.step()

        return train_predicted_loss.item()

    def update_test(self, **kwargs):
        """
        Computes the loss on the test dataset without updating the model's weights.

        Parameters:
            kwargs (dict): Contains 'images' and 'masks', both of type torch.Tensor, representing
                           the input images and their corresponding ground truth masks.

        Returns:
            float: The testing loss for the current step.
        """
        test_predicted_masks = self.model(kwargs["images"])

        test_predicted_loss = self.criterion(test_predicted_masks, kwargs["masks"])

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
        if self.display == True:
            print(
                "Epochs: [{}/{}] - train_loss: [{:.5f}] - test_loss: [{:.5f}]".format(
                    kwargs["epoch"],
                    kwargs["epochs"],
                    kwargs["train_loss"],
                    kwargs["test_loss"],
                )
            )
        elif self.display == False:
            print(
                "Epochs - {}/{} is completed".format(kwargs["epoch"], kwargs["epochs"])
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

        for epoch in tqdm(range(self.epochs)):
            total_train_loss = list()
            total_test_loss = list()

            for images, masks in self.train_dataloader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                total_train_loss.append(self.update_train(images=images, masks=masks))

            for images, masks in self.test_dataloader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                total_test_loss.append(self.update_test(images=images, masks=masks))

            try:
                self.saved_checkpoints(epoch=epoch + 1)

                self.history["train_loss"].append(np.mean(total_train_loss))
                self.history["test_loss"].append(np.mean(total_test_loss))

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
                        nrow=4,
                        normalize=True,
                    )
                else:
                    raise Exception("Train images path not found.".capitalize())

            finally:
                self.show_progress(
                    epoch=epoch + 1,
                    epochs=self.epochs,
                    train_loss=np.mean(total_train_loss),
                    test_loss=np.mean(total_test_loss),
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
        "--loss", type=str, default=None, help="Loss function".capitalize()
    )
    parser.add_argument(
        "--attentionUNet", type=bool, default=False, help="Attention UNet".capitalize()
    )
    parser.add_argument(
        "--display", type=bool, default=True, help="Display progress".capitalize()
    )
    parser.add_argument("--device", type=str, default="mps", help="Device".capitalize())
    parser.add_argument(
        "--smooth", type=float, default=0.01, help="Smooth value".capitalize()
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Alpha value".capitalize()
    )
    parser.add_argument(
        "--gamma", type=float, default=2, help="Gamma value".capitalize()
    )
    parser.add_argument(
        "--is_l1", type=bool, default=False, help="L1 value".capitalize()
    )
    parser.add_argument(
        "--is_l2", type=bool, default=False, help="L2 value".capitalize()
    )
    parser.add_argument(
        "--is_elastic", type=bool, default=False, help="Elastic Transform".capitalize()
    )
    parser.add_argument(
        "--is_weight_clip", type=bool, default=False, help="Weight Clip".capitalize()
    )
    parser.add_argument("--train", action="store_true", help="Train model".capitalize())

    args = parser.parse_args()

    if args.train:
        trainer = Trainer(
            epochs=args.epochs,
            lr=args.lr,
            loss=args.loss,
            is_attentionUNet=args.attentionUNet,
            is_l1=args.is_l1,
            is_l2=args.is_l2,
            is_elastic=args.is_elastic,
            is_weight_clip=args.is_weight_clip,
            alpha=args.alpha,
            gamma=args.gamma,
            display=args.display,
            device=args.device,
            smooth=args.smooth,
        )

        trainer.train()
    else:
        raise Exception("Train flag is not set.".capitalize())
