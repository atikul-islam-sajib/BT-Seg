# Dice Loss for Image Segmentation

## Overview

The `DiceLoss` class implements the Dice Loss, a common metric used for evaluating the performance of image segmentation tasks, especially binary segmentation. Dice Loss measures the overlap between two samples, with a value of 1 indicating perfect overlap. It is particularly useful in medical image segmentation where precision is crucial.

## Installation

This implementation requires Python and PyTorch. Ensure you have Python installed on your system, then install PyTorch by following the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/).

## Dependencies

- Python 3.x
- PyTorch

You can install PyTorch with pip:

```sh
pip install torch torchvision
```

## Usage

### Quick Start

Here's a quick example to get you started with using the Dice Loss in your segmentation projects:

```python
import torch
from dice_loss import DiceLoss  # Assuming the class is saved in dice_loss.py

dice_loss = DiceLoss(smooth=1e-6)
predicted = torch.randn(10, 1, 256, 256)
actual = torch.randn(10, 1, 256, 256)
loss = dice_loss(predicted, actual)
print(f"Dice Loss: {loss.item()}")
```

### Class Parameters

`DiceLoss` initialization parameters:

| Parameter | Type  | Default | Description                                   |
| --------- | ----- | ------- | --------------------------------------------- |
| smooth    | float | 1e-6    | Smoothing factor to prevent division by zero. |

### CLI Arguments

If using the command line interface for testing:

| Argument   | Type  | Default | Description                                                      |
| ---------- | ----- | ------- | ---------------------------------------------------------------- |
| `--smooth` | float | 1e-6    | Smooth value to avoid division by zero in Dice Loss calculation. |

### Command Line Interface

The script also includes a simple CLI for testing the Dice Loss computation with a predefined smoothing factor. Run the following command for a demonstration:

```sh
python dice_loss.py --smooth 1e-6
```
