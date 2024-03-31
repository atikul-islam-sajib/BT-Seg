import sys
import os
import logging
import argparse
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="./logs/U-Net.log",
)

sys.path.append("src/")

from encoder import Encoder
from decoder import Decoder


class UNet(nn.Module):
    """
    Implements the U-Net architecture for image segmentation tasks.

    The U-Net model is composed of an encoder (downsampling path),
    a bottleneck, and a decoder (upsampling path) with skip connections.

    ### Parameters

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | N/A       | N/A  | This class does not accept parameters in its constructor. |

    ### Attributes

    | Attribute           | Type        | Description |
    |---------------------|-------------|-------------|
    | `encoder_layer1`    | `Encoder`   | The first encoder layer. |
    | `encoder_layer2`    | `Encoder`   | The second encoder layer. |
    | `encoder_layer3`    | `Encoder`   | The third encoder layer. |
    | `encoder_layer4`    | `Encoder`   | The fourth encoder layer. |
    | `bottom_layer`      | `Encoder`   | The bottom layer (bottleneck). |
    | `decoder_layer1`    | `Decoder`   | The first decoder layer. |
    | `decoder_layer2`    | `Decoder`   | The second decoder layer. |
    | `decoder_layer3`    | `Decoder`   | The third decoder layer. |
    | `decoder_layer4`    | `Decoder`   | The fourth decoder layer. |
    | `final_layer`       | `Sequential`| The final layer applying convolution and sigmoid activation. |

    ### Example

    ```python
    model = UNet()
    input = torch.randn(1, 3, 256, 256)  # Example input
    output = model(input)
    print(output.shape)  # Expected output shape: [1, 1, 256, 256]
    ```

    Note: The input tensor's dimensions should match the expected dimensions of the U-Net model, typically `[batch_size, channels, height, width]`.
    """

    def __init__(self):
        super(UNet, self).__init__()
        self.encoder_layer1 = Encoder(in_channels=3, out_channels=64)
        self.encoder_layer2 = Encoder(in_channels=64, out_channels=128)
        self.encoder_layer3 = Encoder(in_channels=128, out_channels=256)
        self.encoder_layer4 = Encoder(in_channels=256, out_channels=512)
        self.bottom_layer = Encoder(in_channels=512, out_channels=1024)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.intermediate_layer1 = Encoder(in_channels=1024, out_channels=512)
        self.intermediate_layer2 = Encoder(in_channels=512, out_channels=256)
        self.intermediate_layer3 = Encoder(in_channels=256, out_channels=128)
        self.intermediate_layer4 = Encoder(in_channels=128, out_channels=64)

        self.decoder_layer1 = Decoder(in_channels=1024, out_channels=512)
        self.decoder_layer2 = Decoder(in_channels=512, out_channels=256)
        self.decoder_layer3 = Decoder(in_channels=256, out_channels=128)
        self.decoder_layer4 = Decoder(in_channels=128, out_channels=64)

        self.final_layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1), nn.Sigmoid()
        )

    def forward(self, x):
        """
        Defines the forward pass of the U-Net model.

        The method processes the input through consecutive encoder layers, followed by a bottom layer, and then through consecutive decoder layers. Skip connections are used to concatenate encoder outputs with corresponding decoder inputs for preserving spatial information lost during downsampling.

        ### Parameters

        | Parameter | Type       | Description                       |
        |-----------|------------|-----------------------------------|
        | `x`       | `torch.Tensor` | The input tensor of shape `[batch_size, channels, height, width]`. |

        ### Returns

        | Type         | Description                                         |
        |--------------|-----------------------------------------------------|
        | `torch.Tensor` | The output tensor of shape `[batch_size, 1, height, width]` after sigmoid activation. |

        ### Process

        1. **Encoder Path**: The input is processed through four encoder layers, with max pooling between each layer to reduce spatial dimensions.
        2. **Bottom Layer**: Acts as a bottleneck, further processing the data from the last encoder layer.
        3. **Decoder Path**: Data from the bottom layer is upsampled and concatenated with corresponding encoder outputs (skip connections) then passed through decoder layers.
        4. **Final Output**: The last decoder output is passed through a final layer to produce the segmentation map.

        ### Example

        ```python
        # Assuming an instance `model` of UNet and a 4D input tensor `input`.
        output = model(input)
        # `output` is now a tensor with the same height and width as `input`, and a single channel representing the segmentation map.
        ```

        This method implements the core functionality of the U-Net architecture, making it suitable for various segmentation tasks.
        """
        # Encoder path
        enc1_out = self.encoder_layer1(x)
        pooled_enc1 = self.max_pool(enc1_out)

        enc2_out = self.encoder_layer2(pooled_enc1)
        pooled_enc2 = self.max_pool(enc2_out)

        enc3_out = self.encoder_layer3(pooled_enc2)
        pooled_enc3 = self.max_pool(enc3_out)

        enc4_out = self.encoder_layer4(pooled_enc3)
        pooled_enc4 = self.max_pool(enc4_out)

        bottom_out = self.bottom_layer(pooled_enc4)

        # Decoder path
        dec1_input = self.decoder_layer1(bottom_out, enc4_out)
        dec1_out = self.intermediate_layer1(dec1_input)

        dec2_input = self.decoder_layer2(dec1_out, enc3_out)
        dec2_out = self.intermediate_layer2(dec2_input)

        dec3_input = self.decoder_layer3(dec2_out, enc2_out)
        dec3_out = self.intermediate_layer3(dec3_input)

        dec4_input = self.decoder_layer4(dec3_out, enc1_out)
        dec4_out = self.intermediate_layer4(dec4_input)

        # Final output
        final_output = self.final_layer(dec4_out)

        return final_output

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
    parser = argparse.ArgumentParser(description="UNet model".title())
    parser.add_argument(
        "--unet", action="store_true", help="Run UNet model".capitalize()
    )
    args = parser.parse_args()

    if args.unet:
        model = UNet()
        logging.info(model)
    else:
        raise ValueError("Use the appropriate flag to run the model".capitalize())
