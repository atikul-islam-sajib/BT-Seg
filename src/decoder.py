import sys
import os
import logging
import argparse
from collections import OrderedDict
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="./logs/decoder.log",
)


class Decoder(nn.Module):
    """
    To format the documentation for the `Decoder` class to be compatible with MkDocs, including the use of tables for parameters and arguments, follow the structured template below. This template covers the class description, methods, parameters, and command-line usage.

    This document provides detailed information about the `Decoder` class and its application within a script. The `Decoder` class is designed to create a convolutional neural network decoder block, typically used in conjunction with an encoder in U-Net architectures for image segmentation tasks.

    ## Decoder Class

    The `Decoder` class is responsible for initializing and constructing a decoder block with transposed convolutional layers for upscaling feature maps and merging skip connections.

    ### __init__(self, in_channels=None, out_channels=None)

    Initializes the `Decoder` class with specified input and output channels for the decoder block.

    | Parameter     | Type  | Description                               |
    |---------------|-------|-------------------------------------------|
    | `in_channels` | int   | The number of input channels.             |
    | `out_channels`| int   | The number of output channels.            |

    ### decoder_block(self)

    Builds the decoder block using transposed convolutional layers.

    | Returns       | Type          | Description                       |
    |---------------|---------------|-----------------------------------|
    | `model`       | nn.Sequential | The sequential model of the decoder block. |

    ### forward(self, x, skip_info)

    Defines the forward pass for the decoder block, including the concatenation of the input tensor with skip connection information.

    | Parameter  | Type         | Description                              |
    |------------|--------------|------------------------------------------|
    | `x`        | torch.Tensor | The input tensor to the decoder block.   |
    | `skip_info`| torch.Tensor | The tensor containing skip connection information to be concatenated with the output of the decoder block. |

    | Returns    | Type         | Description                             |
    |------------|--------------|-----------------------------------------|
    | `output`   | torch.Tensor | The output tensor from the decoder block after concatenation with the skip information. |

    ## Script Usage

    The script can be executed with command-line arguments to initiate the decoder block for a U-Net architecture.

    ### Command-line Arguments

    | Argument    | Type    | Description                            |
    |-------------|---------|----------------------------------------|
    | `--decoder` | flag    | Triggers the creation of the decoder block. |

    ### Example

    ```bash
    python script.py --decoder
    ```
    """

    def __init__(self, in_channels=None, out_channels=None):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.model = self.decoder_block()

    def decoder_block(self):
        layers = OrderedDict()
        layers["deconv1"] = nn.ConvTranspose2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=2,
            stride=2,
        )
        return nn.Sequential(layers)

    def forward(self, x, skip_info):
        if x is not None and skip_info is not None:
            return torch.cat((self.model(x), skip_info), dim=1)
        else:
            raise ValueError("Input and skip_info cannot be None".capitalize())

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
        description="Define the Decoder block for U-Net".title()
    )
    parser.add_argument(
        "--decoder", action="store_true", help="Decoder block".capitalize()
    )
    args = parser.parse_args()

    if args.decoder:
        decoder = Decoder(in_channels=1024, out_channels=512)
        logging.info(decoder)
    else:
        raise ValueError("Define the arguments for the decoder".capitalize())
