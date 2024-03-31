import sys
import os
import logging
import argparse
from collections import OrderedDict
import torch
import torch.nn as nn


class AttentionBlock(nn.Module):
    def __init__(self, in_channels=None, out_channels=None):
        super(AttentionBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.W_gate = self.W_gate_block()
        self.W_x = self.W_x_block()
        self.psi = self.pis_block()
        self.relu = nn.ReLU(inplace=True)

    def W_gate_block(self):
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

    def pis_block(self):
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
        transformed_input = self.W_gate(x)

        if skip_info is not None:
            transformed_skip = self.W_x(skip_info)

        merged_features = self.relu(
            torch.cat((transformed_input, transformed_skip), dim=1)
        )
        attention_weights = self.psi(merged_features)

        return transformed_skip * attention_weights


if __name__ == "__main__":
    out_result = torch.randn(64, 512, 16, 16)
    skip_info = torch.randn(64, 512, 16, 16)

    attention = AttentionBlock(in_channels=512, out_channels=512)
    assert attention(out_result, skip_info).shape == (64, 512, 16, 16)
