import sys
import os
import logging
import unittest
import torch

sys.path.append("src/")

from utils import load
from config import RAW_PATH, PROCESSED_PATH
from UNet import UNet
from encoder import Encoder
from decoder import Decoder


class UnitTest(unittest.TestCase):
    """
    Defines a suite of unit tests for verifying the integrity and correctness of the
    dataset loading functionality and dataset sizes for training and testing.
    """

    def setUp(self):
        """
        Set up the test environment by loading the train and test dataloaders from
        the PROCESSED_PATH.
        """
        self.train_dataloader = load(
            os.path.join(PROCESSED_PATH, "train_dataloader.pkl")
        )
        self.test_dataloader = load(
            os.path.join(PROCESSED_PATH, "test_data_loader.pkl")
        )
        self.encoder = Encoder(in_channels=3, out_channels=64)
        self.decoder = Decoder(in_channels=1024, out_channels=512)
        self.model = UNet()

    def test_train_dataset_size(self):
        """
        Test that the dimensions of the data and label tensors in the training dataset
        are as expected.
        """
        data, label = next(iter(self.train_dataloader))
        self.assertEqual(data, torch.Size([4, 3, 256, 256]))
        self.assertEqual(label, torch.Size([4, 1, 256, 256]))

    def test_val_dataset_size(self):
        """
        Test that the dimensions of the data and label tensors in the validation dataset
        are as expected.
        """
        data, label = next(iter(self.test_dataloader))
        self.assertEqual(data, torch.Size([24, 3, 256, 256]))
        self.assertEqual(label, torch.Size([24, 1, 256, 256]))

    def test_quantity_train_dataset(self):
        """
        Test that the total number of samples in the training dataset is as expected.
        """
        self.assertEqual(sum(data.size(0) for data in self.train_dataloader), 3256)

    def test_quantity_val_dataset(self):
        """
        Test that the total number of samples in the validation dataset is as expected.
        """
        self.assertEqual(sum(data.size(0) for data in self.test_dataloader), 3256)

    def test_total_params(self):
        """
        Test that the total number of trainable parameters in the model is as expected.
        This helps ensure the model architecture has not inadvertently changed.
        """
        self.assertEqual(
            sum(params.numel() for params in self.model.parameters()), 31037633
        )

    def test_encoder_block(self):
        """
        Test that the encoder block of the model produces output of the expected shape
        when fed with random noise input. This verifies the encoder's functionality and output dimensions.
        """
        noise_data = torch.randn(64, 3, 256, 256)
        self.assertEqual(self.encoder(noise_data).shape, torch.Size([64, 64, 256, 256]))

    def test_decoder_block(self):
        """
        Test that the decoder block of the model, when given random noise input and
        skip connection information, produces output of the expected shape. This checks
        the decoder's integration and functionality with respect to handling inputs and skip connections.
        """
        noise_data = torch.randn(64, 1024, 16, 16)
        skip_info = torch.randn(64, 512, 32, 32)
        self.assertEqual(
            self.decoder(noise_data, skip_info).shape,
            torch.Size([64, 1024, 32, 32]),
        )


if __name__ == "__main__":
    unittest.main()
