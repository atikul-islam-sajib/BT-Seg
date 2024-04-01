# Jaccard Loss (IoU Loss) for Image Segmentation

## Overview

The `JaccardLoss` class implements the Jaccard Loss, also known as Intersection over Union (IoU) Loss, a prominent loss function for segmentation tasks. It is particularly effective in scenarios where the object of interest occupies a significantly smaller portion of the image compared to the background. The Jaccard Index, or IoU, measures the similarity between two sets and is a vital metric for the evaluation of image segmentation accuracy.

## Installation

This implementation requires Python and PyTorch. Ensure Python is installed on your system, and then install PyTorch by following the official instructions at [PyTorch's website](https://pytorch.org/get-started/locally/).

## Dependencies

- Python 3.x
- PyTorch

To install PyTorch, you can use pip:

```sh
pip install torch torchvision
```

## Usage

### Instantiation

Initialize `JaccardLoss` with an optional smoothing factor to avoid division by zero errors:

```python
from jaccard_loss import JaccardLoss  # Assuming the class is in jaccard_loss.py

jaccard_loss = JaccardLoss(smooth=1e-6)
```

### Loss Calculation

Use the loss in your training loop to compute the similarity between the predicted and actual segmentation maps:

```python
model = UNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Example tensors for predicted and actual segmentation maps
predicted = model(images)  # Your model's predictions
actual = masks  # Ground truth masks

loss = jaccard_loss(predicted, actual)
loss.backward()
optimizer.step()
```

### Command Line Testing

A simple CLI is provided to test the Jaccard Loss calculation directly:

```sh
python jaccard_loss.py --smooth 1e-6
```

## Parameters and Arguments

### Class Parameters

`JaccardLoss` initialization parameter:

| Parameter | Type  | Default | Description                                   |
| --------- | ----- | ------- | --------------------------------------------- |
| smooth    | float | 1e-6    | Smoothing factor to prevent division by zero. |

### CLI Arguments

Arguments for the command-line interface:

| Argument   | Type  | Default | Description                                    |
| ---------- | ----- | ------- | ---------------------------------------------- |
| `--smooth` | float | 1e-6    | Smoothing factor for Jaccard Loss calculation. |
