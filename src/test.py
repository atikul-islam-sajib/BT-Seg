import sys
import os
import logging
import argparse
import imageio
import matplotlib.pyplot as plt
import torch

from config import (
    BEST_MODEL_PATH,
    PROCESSED_PATH,
    TEST_IMAGE_PATH,
    TRAIN_IMAGES_PATH,
    GIF_PATH,
)
from utils import device_init, load
from UNet import UNet
from AttentionUNet import AttentionUNet


class Charts:
    """
    A class responsible for visualizing model performance through charts and generating GIFs of training progress.

    | Attribute | Type   | Description                                  |
    |-----------|--------|----------------------------------------------|
    | device    | str    | The device on which to perform computations. |

    ## Methods

    - `__init__(self, device="mps")`
        Initializes the Charts object with a specified computing device.

    - `select_best_model(self)`
        Selects the best model based on saved checkpoints.

    - `obtain_dataloader(self)`
        Loads the test dataset.

    - `data_normalized(self, **kwargs)`
        Normalizes the data to be plotted.

    - `plot_data_comparison(self)`
        Plots a comparison between original images, their masks, and the predicted masks.

    - `generate_gif(self)`
        Generates a GIF from training images to visualize progress.

    - `test(self)`
        Executes the test phase, including model evaluation and visualization.

    ## Examples

    ```python
    charts = Charts(device="cuda")
    charts.test()
    ```
    """

    def __init__(self, device="mps", is_attentionUNet=None):
        """
        Initializes the Charts object.

        | Parameter | Type | Default | Description                                   |
        |-----------|------|---------|-----------------------------------------------|
        | device    | str  | "mps"   | The device on which to perform computations.  |
        """
        self.device = device_init(device=device)
        self.is_attentionUNet = is_attentionUNet

    def select_best_model(self):
        """
        Selects and loads the best performing model based on saved checkpoints.

        | Returns  | Type             | Description                              |
        |----------|------------------|------------------------------------------|
        | model    | torch.nn.Module  | The loaded PyTorch model in evaluation mode. |

        | Raises   |
        |----------|
        | Exception| If the best model is not found.                              |
        """
        if os.path.exists(BEST_MODEL_PATH):
            return torch.load(os.path.join(BEST_MODEL_PATH, "best_model.pth"))
        else:
            raise Exception("Best model not found.".capitalize())

    def obtain_dataloader(self):
        """
        Loads the test dataloader from the processed data path.

        | Returns  | Type                        | Description                   |
        |----------|-----------------------------|-------------------------------|
        | dataloader | torch.utils.data.DataLoader | The test dataset dataloader. |

        | Raises   |
        |----------|
        | Exception| If the processed data is not found.                         |
        """
        if os.path.exists(PROCESSED_PATH):
            return load(os.path.join(PROCESSED_PATH, "test_dataloader.pkl"))
        else:
            raise Exception(
                "Processed data not found. Please run the preprocessing notebook.".capitalize()
            )

    def data_normalized(self, **kwargs):
        """
        Normalizes input data to a range of [0, 1].

        | Parameters | Type          | Description                            |
        |------------|---------------|----------------------------------------|
        | **kwargs   | dict          | Keyword arguments including 'data', a torch.Tensor to normalize. |

        | Returns    | Type          | Description                            |
        |------------|---------------|----------------------------------------|
        | normalized_data | torch.Tensor | The normalized data tensor.           |
        """
        return (kwargs["data"] - kwargs["data"].min()) / (
            kwargs["data"].max() - kwargs["data"].min()
        )

    def plot_data_comparison(self):
        """
        Plots and saves a comparison between original images, their masks, and the predicted masks by the model.

        | Raises    |
        |-----------|
        | Exception | If there is an issue saving the plot.                        |
        """
        plt.figure(figsize=(40, 35))

        images, original_masks = next(iter(self.obtain_dataloader()))
        images = images.to(self.device)
        original_masks = original_masks.to(self.device)
        predicted_masks = self.model(images)

        for index, image in enumerate(images):
            plt.subplot(3 * 4, 3 * 4, 3 * index + 1)
            image = image.permute(1, 2, 0).detach().cpu().numpy()
            image = self.data_normalized(data=image)
            plt.imshow(image)
            plt.title("Image")
            plt.axis("off")

            plt.subplot(3 * 4, 3 * 4, 3 * index + 2)
            mask = original_masks[index].permute(1, 2, 0).detach().cpu().numpy()
            mask = self.data_normalized(data=mask)
            plt.imshow(mask)
            plt.title("Masks")
            plt.axis("off")

            plt.subplot(3 * 4, 3 * 4, 3 * index + 3)
            pred_mask = predicted_masks[index].permute(1, 2, 0).detach().cpu().numpy()
            pred_mask = self.data_normalized(data=pred_mask)
            plt.imshow(pred_mask)
            plt.title("Predicted")
            plt.axis("off")

        try:
            if os.path.exists(TEST_IMAGE_PATH):
                plt.tight_layout()
                plt.savefig(os.path.join(TEST_IMAGE_PATH, "result.png"))
            else:
                os.makedirs(TEST_IMAGE_PATH)
        except Exception as e:
            print("The exception in the section # {}".format(e).capitalize())
        else:
            plt.show()

    def generate_gif(self):
        """
        Generates a GIF from saved training images to visualize the training progress.

        | Raises    |
        |-----------|
        | Exception | If the training images path is not found.                    |
        """
        if os.path.exists(TRAIN_IMAGES_PATH) and os.path.exists(GIF_PATH):
            images = [
                imageio.imread(os.path.join(TRAIN_IMAGES_PATH, image))
                for image in os.listdir(TRAIN_IMAGES_PATH)
            ]
            imageio.mimsave(os.path.join(GIF_PATH, "train_masks.gif"), images, "GIF")
        else:
            raise Exception("Train images path not found.".capitalize())

    def test(self):
        """
        Conducts the test by evaluating the model and generating visualizations of its performance.

        | Raises    |
        |-----------|
        | Exception | If there is an exception during the test phase.               |
        """
        if self.is_attentionUNet:
            self.model = AttentionUNet().to(self.device)
        else:
            self.model = UNet().to(self.device)

        try:
            self.model.load_state_dict(self.select_best_model())
        except Exception as e:
            print("The exception in the section # {}".format(e).capitalize())
        try:
            self.plot_data_comparison()
        except Exception as e:
            print("The exception in the section # {}".format(e).capitalize())
        try:
            self.generate_gif()
        except Exception as e:
            print("The exception in the section # {}".format(e).capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing the model.".title())
    parser.add_argument(
        "--device", type=str, default="mps", help="Device to run the model on."
    )

    args = parser.parse_args()

    if args.device:
        charts = Charts(device="mps")
        charts.test()
    else:
        raise Exception("Device not specified.".capitalize())
