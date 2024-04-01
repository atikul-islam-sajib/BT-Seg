# Tversky Loss for Image Segmentation

## Overview

The `TverskyLoss` class implements the Tversky Loss function, a flexible and generalization of the Dice Loss that allows for the adjustment of the importance between false positives and false negatives. This adjustment capability makes it particularly useful for segmentation tasks with imbalanced datasets.

## Installation

This implementation requires Python and PyTorch. If you haven't already, install Python on your system and follow the [PyTorch official installation guide](https://pytorch.org/get-started/locally/) to set up PyTorch.

## Dependencies

- Python 3.x
- PyTorch

To install PyTorch via pip, run:

```sh
pip install torch torchvision
```

## Usage

### Instantiating Tversky Loss

Create an instance of `TverskyLoss` with an optional smoothing factor to prevent division by zero in the loss calculation:

```python
from tversky_loss import TverskyLoss  # Assuming the class is saved in tversky_loss.py

tversky_loss = TverskyLoss(smooth=1e-6)
```

### Applying Tversky Loss

Incorporate `TverskyLoss` in your training loop to handle imbalanced segmentation data:

```python
model = UNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

predicted = model(input_data)  # input_data is your batch of input images
actual = target_data  # target_data is your batch of target segmentation masks

loss = tversky_loss(predicted, actual)
loss.backward()
optimizer.step()
```

### Command Line Interface

A simple CLI is included for testing the Tversky Loss function directly:

```sh
python tversky_loss.py --smooth 1e-6
```

## Parameters and Arguments

### Class Parameters

`TverskyLoss` initialization parameters:

| Parameter | Type  | Default | Description                                   |
| --------- | ----- | ------- | --------------------------------------------- |
| smooth    | float | 1e-6    | Smoothing factor to prevent division by zero. |

### CLI Arguments

Arguments for testing `TverskyLoss` via the command line:

| Argument   | Type  | Default | Description                                    |
| ---------- | ----- | ------- | ---------------------------------------------- |
| `--smooth` | float | 1e-6    | Smoothing factor for Tversky Loss calculation. |
