# DiceBCE Loss for Binary Segmentation

## Overview

The `DiceBCE` class combines Dice Loss and Binary Cross-Entropy (BCE) Loss, offering a robust loss function for binary segmentation tasks. This hybrid approach takes advantage of Dice Loss's effectiveness in handling class imbalance and the straightforward approach of BCE Loss to measure the difference between predicted and actual segmentation maps.

## Installation

Ensure Python and PyTorch are installed on your system. If not, follow the [PyTorch installation guide](https://pytorch.org/get-started/locally/) to get started.

## Dependencies

- Python 3.x
- PyTorch

PyTorch can be installed via pip:

```sh
pip install torch torchvision
```

## Usage

### Creating the Loss Function

Instantiate `DiceBCE` with an optional smoothing factor to avoid division by zero in the Dice Loss calculation. This factor is particularly useful when the predicted and actual masks might have no overlap.

```python
from dice_bce_loss import DiceBCE  # Assuming you've saved the class in dice_bce_loss.py

criterion = DiceBCE(smooth=1e-6)
```

### Training with DiceBCE

Use `DiceBCE` as the loss function in your training loop. Here's a minimal example assuming you have a model for binary segmentation:

```python
model = UNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

predicted = model(input_images)  # input_images should be your batch of input images
actual = target_masks  # target_masks should be the corresponding ground truth masks

loss = criterion(predicted, actual)
loss.backward()
optimizer.step()
```

### CLI Testing

A simple command-line interface is included for testing the DiceBCE loss calculation with custom smoothing factors:

```sh
python dice_bce_loss.py --smooth 1e-6
```

## Parameters and Arguments

### Class Parameters

`DiceBCE` initialization parameters:

| Parameter | Type  | Default | Description                                                        |
| --------- | ----- | ------- | ------------------------------------------------------------------ |
| smooth    | float | 1e-6    | Smoothing factor for Dice calculation to prevent division by zero. |

### CLI Arguments

When using the CLI for testing:

| Argument   | Type  | Default | Description                                 |
| ---------- | ----- | ------- | ------------------------------------------- |
| `--smooth` | float | 1e-6    | Smoothing factor for Dice loss calculation. |
