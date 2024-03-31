import sys
import os
import logging
import argparse
from collections import OrderedDict
import torch
import torch.nn as nn


class AttentionBlock(nn.Module):
    """
    Implements an attention block for convolutional neural networks.

    | Parameter     | Type | Description                      |
    |---------------|------|----------------------------------|
    | in_channels   | int  | Number of input channels.        |
    | out_channels  | int  | Number of output channels.       |

    Methods:
    - `W_gate_block()`: Initializes the gating signal transformation.
    - `W_x_block()`: Initializes the skip connection transformation.
    - `psi_block()`: Initializes the attention coefficients calculation.
    - `forward(x, skip_info)`: Applies the attention mechanism.

    Example usage:
    ```python
    attention_block = AttentionBlock(in_channels=512, out_channels=512)
    output = attention_block(input_tensor, skip_info_tensor)
    ```
    """

    def __init__(self, in_channels=None, out_channels=None):
        super(AttentionBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.W_gate = self.W_gate_block()
        self.W_x = self.W_x_block()
        self.psi = self.psi_block()
        self.relu = nn.ReLU(inplace=True)

    def W_gate_block(self):
        """
        Constructs the W_gate layer sequence.

        | Returns      | Type                | Description                       |
        |--------------|---------------------|-----------------------------------|
        | nn.Sequential | PyTorch Sequential | Sequential model for gate signal. |
        """
        layers = OrderedDict()

        layers["W_gate_conv"] = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        layers["W_gate_batch_norm"] = nn.BatchNorm2d(self.out_channels)

        return nn.Sequential(layers)

    def W_x_block(self):
        """
        Constructs the W_x layer sequence.

        | Returns      | Type                | Description                        |
        |--------------|---------------------|------------------------------------|
        | nn.Sequential | PyTorch Sequential | Sequential model for skip signal. |
        """
        layers = OrderedDict()

        layers["W_x_conv"] = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        layers["W_x_batch_norm"] = nn.BatchNorm2d(self.out_channels)

        return nn.Sequential(layers)

    def psi_block(self):
        """
        Constructs the psi layer sequence for attention coefficient calculation.

        | Returns      | Type                | Description                           |
        |--------------|---------------------|---------------------------------------|
        | nn.Sequential | PyTorch Sequential | Sequential model for attention psi. |
        """
        layers = OrderedDict()

        layers["psi_conv"] = nn.Conv2d(
            in_channels=self.in_channels * 2,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        layers["psi_batch_norm"] = nn.BatchNorm2d(1)
        layers["psi_sigmoid"] = nn.Sigmoid()

        return nn.Sequential(layers)

    def forward(self, x, skip_info):
        """
        Forward pass of the AttentionBlock.

        | Parameter  | Type         | Description                    |
        |------------|--------------|--------------------------------|
        | x          | torch.Tensor | Input feature map.             |
        | skip_info  | torch.Tensor | Feature map from skip connection. |

        | Returns    | Type         | Description                     |
        |------------|--------------|---------------------------------|
        | torch.Tensor | torch.Tensor | Attention-modulated feature map. |
        """
        transformed_input = self.W_gate(x)

        if skip_info is not None:
            transformed_skip = self.W_x(skip_info)

        merged_features = self.relu(
            torch.cat((transformed_input, transformed_skip), dim=1)
        )
        attention_weights = self.psi(merged_features)

        return transformed_skip * attention_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AttentionBlock Test".capitalize())
    parser.add_argument(
        "--attention_block", action="store_true", default="AttentionBlock"
    )
    args = parser.parse_args()

    if args.attention_block:
        out_result = torch.randn(64, 512, 16, 16)
        skip_info = torch.randn(64, 512, 16, 16)

        attention = AttentionBlock(in_channels=512, out_channels=512)
        print(attention(out_result, skip_info).shape)
    else:
        raise Exception(
            "Arguments should be provided in an appropriate manner".capitalize()
        )
