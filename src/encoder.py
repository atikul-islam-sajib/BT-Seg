import sys
import argparse
from collections import OrderedDict
import torch.nn as nn

sys.path.append("src/")


class Encoder(nn.Module):
    """
    This document provides an overview of the `Encoder` class and its usage in the script. The `Encoder` class is designed for creating a convolutional neural network encoder block, primarily used in U-Net architectures for tasks such as image segmentation.

    The `Encoder` class initializes and constructs an encoder block with convolutional, batch normalization, and ReLU layers.

    ### __init__(self, in_channels=None, out_channels=None)

    Initializes the `Encoder` class with the specified input and output channels.

    | Parameter     | Type  | Description                              |
    |---------------|-------|------------------------------------------|
    | `in_channels` | int   | The number of input channels.            |
    | `out_channels`| int   | The number of output channels.           |

    ### encoder_block(self)

    Constructs the encoder block with convolutional, ReLU, and batch normalization layers.

    | Returns       | Type          | Description                      |
    |---------------|---------------|----------------------------------|
    | `model`       | nn.Sequential | The sequential model of the encoder block. |

    ### forward(self, x)

    Defines the forward pass for the encoder block.

    | Parameter | Type        | Description                       |
    |-----------|-------------|-----------------------------------|
    | `x`       | torch.Tensor| The input tensor to the encoder block. |

    | Returns   | Type        | Description                        |
    |-----------|-------------|------------------------------------|
    | `output`  | torch.Tensor| The output tensor from the encoder block. |

    ## Script Usage

    The script can be run with command-line arguments to define the encoder block for U-Net.

    ### Command-line Arguments

    | Argument    | Type    | Description                          |
    |-------------|---------|--------------------------------------|
    | `--encoder` | flag    | Activates the encoder block creation.|

    ### Example

    ```bash
    python script.py --encoder
    ```
    """

    def __init__(self, in_channels=None, out_channels=None):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.model = self.encoder_block()

    def encoder_block(self):
        layers = OrderedDict()
        layers["conv1"] = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        layers["relu1"] = nn.ReLU(inplace=True)
        layers["conv2"] = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        layers["batch_norm1"] = nn.BatchNorm2d(self.out_channels)
        layers["relu2"] = nn.ReLU(inplace=True)

        return nn.Sequential(layers)

    def forward(self, x):
        return self.model(x) if x is not None else None

    @staticmethod
    def total_params(model):
        """
        Calculates the total number of parameters in a given PyTorch model.

        This function iterates over all parameters in the model, counting the total number of
        elements (i.e., the product of the size of each dimension of the parameter). It is useful for
        getting a quick understanding of the model's complexity and size.

        Parameters:
        - model (torch.nn.Module): The PyTorch model whose parameters are to be counted.

        Returns:
        - int: The total number of parameters in the model.
        """
        return sum(params.numel() for params in model.parameters())


if __name__ == "__main__":
    parser = parser = argparse.ArgumentParser(
        description="Define the Encoder block for U-Net".title()
    )
    parser.add_argument(
        "--encoder", action="store_true", help="Encoder block".capitalize()
    )
    args = parser.parse_args()
    if args.encoder:
        encoder = Encoder(in_channels=3, out_channels=64)

        print("Total parameters in the Encoder block:", Encoder.total_params(encoder))
    else:
        raise ValueError("Define the arguments in an appropriate way".capitalize())
