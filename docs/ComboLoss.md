# Combo Loss for Improved Segmentation

## Overview

The `ComboLoss` class offers a multi-faceted loss function combining Binary Cross-Entropy (BCE) Loss, Focal Loss, and Dice Loss. This approach is aimed at enhancing model performance on segmentation tasks, particularly when dealing with imbalanced datasets. By integrating these loss functions, `ComboLoss` aims to leverage the benefits of both probabilistic loss measures and spatial overlap, while emphasizing difficult-to-classify examples.

## Installation

This implementation requires Python and PyTorch. If not already installed, you can set up Python on your machine and then follow the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Dependencies

- Python 3.x
- PyTorch

PyTorch can be installed using pip:

```sh
pip install torch torchvision
```

## Usage

### Initializing Combo Loss

Instantiate the `ComboLoss` class with the smoothing factor (`smooth`), alpha (`alpha`), and gamma (`gamma`) parameters for customizability:

```python
from combo_loss import ComboLoss  # Assuming the class is saved in combo_loss.py

combo_loss = ComboLoss(smooth=1e-6, alpha=0.5, gamma=0.5)
```

### Example Training Loop

Here's how you can incorporate `ComboLoss` into a training loop for a segmentation model:

```python
model = UNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Assuming `predicted` and `actual` are your model outputs and ground truth, respectively
loss = combo_loss(predicted, actual)
loss.backward()
optimizer.step()
```

### Command Line Testing

A CLI is included for direct testing of the `ComboLoss` with customizable parameters:

```sh
python combo_loss.py --smooth 1e-6 --alpha 0.5 --gamma 0.5
```

## Parameters and Arguments

### Class Parameters

Initialization parameters for `ComboLoss`:

| Parameter | Type  | Default | Description                                               |
| --------- | ----- | ------- | --------------------------------------------------------- |
| smooth    | float | 1e-6    | Smoothing factor for Dice Loss calculation.               |
| alpha     | float | 0.5     | Alpha parameter for Focal Loss.                           |
| gamma     | float | 0.5     | Gamma parameter for Focal Loss to focus on hard examples. |

### CLI Arguments

Command-line arguments for testing `ComboLoss`:

| Argument   | Type  | Default | Description                                   |
| ---------- | ----- | ------- | --------------------------------------------- |
| `--smooth` | float | 1e-6    | Smoothing constant for Dice Loss calculation. |
| `--alpha`  | float | 0.5     | Alpha value for Focal Loss.                   |
| `--gamma`  | float | 0.5     | Gamma value for Focal Loss.                   |
