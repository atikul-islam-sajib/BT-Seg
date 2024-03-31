import sys
import argparse
import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder
from attention_block import AttentionBlock


class AttentionUNet(nn.Module):
    """
    Implements the Attention U-Net architecture for image segmentation.

    This architecture incorporates attention blocks into the traditional U-Net structure to focus on relevant features for segmentation tasks.

    Attributes:

    | Attribute           | Type        | Description                                           |
    |---------------------|-------------|-------------------------------------------------------|
    | `encoder_layer1`    | `Encoder`   | The first encoder layer with 3 input and 64 output channels. |
    | `encoder_layer2`    | `Encoder`   | The second encoder layer with 64 input and 128 output channels. |
    | `encoder_layer3`    | `Encoder`   | The third encoder layer with 128 input and 256 output channels. |
    | `encoder_layer4`    | `Encoder`   | The fourth encoder layer with 256 input and 512 output channels. |
    | `bottom_layer`      | `Encoder`   | The bottom layer (bottleneck) with 512 input and 1024 output channels. |
    | `max_pool`          | `MaxPool2d` | Max pooling layer with kernel size 2 and stride 2 for downsampling. |
    | `intermediate_layer1` | `Encoder` | Intermediate layer post-bottom layer with 1024 input and 512 output channels. |
    | `intermediate_layer2` | `Encoder` | Second intermediate layer with 512 input and 256 output channels. |
    | `intermediate_layer3` | `Encoder` | Third intermediate layer with 256 input and 128 output channels. |
    | `intermediate_layer4` | `Encoder` | Fourth intermediate layer with 128 input and 64 output channels. |
    | `decoder_layer1`    | `Decoder`   | The first decoder layer combining bottom and encoder layer outputs. |
    | `decoder_layer2`    | `Decoder`   | The second decoder layer for upscaling and combining features. |
    | `decoder_layer3`    | `Decoder`   | The third decoder layer for further upscaling and feature combination. |
    | `decoder_layer4`    | `Decoder`   | The fourth decoder layer for final upscaling before the output layer. |
    | `attention_layer1`  | `AttentionBlock` | First attention block for focusing features before the first decoder layer. |
    | `attention_layer2`  | `AttentionBlock` | Second attention block before the second decoder layer. |
    | `attention_layer3`  | `AttentionBlock` | Third attention block before the third decoder layer. |
    | `attention_layer4`  | `AttentionBlock` | Fourth attention block before the final decoder layer. |
    | `final_layer`       | `Sequential`| The final layer applying a 1x1 convolution and sigmoid activation for segmentation output. |

    Methods:
    - `forward(x)`: Defines the forward pass of the model.
    - `total_params(model)`: Static method to calculate total parameters of the model.

    Example usage:
    ```python
    model = AttentionUNet()
    output = model(input_tensor)
    print(AttentionUNet.total_params(model))
    ```

    | Method            | Description                                      |
    |-------------------|--------------------------------------------------|
    | `forward`         | Performs the forward pass of the network.        |
    | `total_params`    | Calculates the total number of parameters.       |
    """

    def __init__(self):
        super(AttentionUNet, self).__init__()

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

        self.attention_layer1 = AttentionBlock(in_channels=512, out_channels=512)
        self.attention_layer2 = AttentionBlock(in_channels=256, out_channels=256)
        self.attention_layer3 = AttentionBlock(in_channels=128, out_channels=128)
        self.attention_layer4 = AttentionBlock(in_channels=64, out_channels=64)

        self.final_layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1), nn.Sigmoid()
        )

    def forward(self, x):
        """
        Defines the forward pass of the Attention U-Net model.

        | Parameter | Type         | Description              |
        |-----------|--------------|--------------------------|
        | x         | torch.Tensor | The input image tensor.  |

        | Returns   | Type         | Description              |
        |-----------|--------------|--------------------------|
        | torch.Tensor | torch.Tensor | The segmented output tensor. |
        """
        # Encoder layers
        encoder_1_output = self.encoder_layer1(x)
        pooled_encoder_1_output = self.max_pool(encoder_1_output)

        encoder_2_output = self.encoder_layer2(pooled_encoder_1_output)
        pooled_encoder_2_output = self.max_pool(encoder_2_output)

        encoder_3_output = self.encoder_layer3(pooled_encoder_2_output)
        pooled_encoder_3_output = self.max_pool(encoder_3_output)

        encoder_4_output = self.encoder_layer4(pooled_encoder_3_output)
        pooled_encoder_4_output = self.max_pool(encoder_4_output)

        bottom_layer_output = self.bottom_layer(pooled_encoder_4_output)
        decoder_1_input = self.decoder_layer1(bottom_layer_output)

        attention_1_output = self.attention_layer1(decoder_1_input, encoder_4_output)
        merged_attention_1_output = torch.cat(
            (attention_1_output, encoder_4_output), dim=1
        )
        decoder_1_output = self.intermediate_layer1(merged_attention_1_output)

        decoder_2_input = self.decoder_layer2(decoder_1_output)
        attention_2_output = self.attention_layer2(decoder_2_input, encoder_3_output)
        merged_attention_2_output = torch.cat(
            (attention_2_output, encoder_3_output), dim=1
        )
        decoder_2_output = self.intermediate_layer2(merged_attention_2_output)

        decoder_3_input = self.decoder_layer3(decoder_2_output)
        attention_3_output = self.attention_layer3(decoder_3_input, encoder_2_output)
        merged_attention_3_output = torch.cat(
            (attention_3_output, encoder_2_output), dim=1
        )
        decoder_3_output = self.intermediate_layer3(merged_attention_3_output)

        decoder_4_input = self.decoder_layer4(decoder_3_output)
        attention_4_output = self.attention_layer4(decoder_4_input, encoder_1_output)
        merged_attention_4_output = torch.cat(
            (attention_4_output, encoder_1_output), dim=1
        )
        decoder_4_output = self.intermediate_layer4(merged_attention_4_output)

        # Final layer
        final_output = self.final_layer(decoder_4_output)

        return final_output

    @staticmethod
    def total_params(model):
        """
        Calculate the total number of parameters in the model.

        | Parameter | Type         | Description            |
        |-----------|--------------|------------------------|
        | model     | nn.Module    | The model to evaluate. |

        | Returns   | Type         | Description            |
        |-----------|--------------|------------------------|
        | int       | int          | Total number of parameters. |
        """
        return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attention UNet")
    parser.add_argument(
        "--AttentionUNet", action="store_true", help="Batch size for training"
    )
    args = parser.parse_args()

    if args.AttentionUNet:
        attentionUNet = AttentionUNet()

        print(AttentionUNet.total_params(attentionUNet))
    else:
        raise Exception(
            "Please provide the command line argument --AttentionUNet".capitalize()
        )
