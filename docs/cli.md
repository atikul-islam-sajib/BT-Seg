# Command Line Interface for UNet Training and Testing

## Overview

This Command Line Interface (CLI) utility facilitates the configuration and execution of training or testing for UNet and Attention UNet models. It provides a versatile set of command-line options for specifying dataset characteristics, training parameters, model choices, and evaluation procedures. The utility is designed to streamline the process of loading data, conducting model training, or performing tests with comprehensive metrics visualization.

## Installation

Ensure Python and necessary dependencies are installed in your environment. This typically includes PyTorch and any specific libraries for data handling or visualization like Matplotlib.

### Command-Line Arguments

The utility supports various arguments to customize its operation:

| Argument           | Description                                 |
| ------------------ | ------------------------------------------- |
| `--image_path`     | Path to the zip file containing the images. |
| `--batch_size`     | Batch size for the dataloader.              |
| `--split_ratio`    | Split ratio for the dataset.                |
| `--image_size`     | Image size for the dataloader.              |
| `--epochs`         | Number of epochs.                           |
| `--lr`             | Learning rate.                              |
| `--loss`           | Loss function to use.                       |
| `--attentionUNet`  | Use Attention UNet model if set.            |
| `--display`        | Display training progress and metrics.      |
| `--device`         | Computation device ('cuda', 'mps', etc.).   |
| `--smooth`         | Smooth value for certain regularization.    |
| `--alpha`          | Alpha value for specific loss functions.    |
| `--gamma`          | Gamma value for specific loss functions.    |
| `--is_l1`          | Enable L1 regularization.                   |
| `--is_l2`          | Enable L2 regularization.                   |
| `--is_elastic`     | Apply elastic transformation to the data.   |
| `--is_weight_clip` | Enable weight clipping.                     |
| `--train`          | Flag to initiate the training process.      |
| `--test`           | Flag to initiate the testing process.       |

### Loss

| Loss Function | Parameters           | CLI Usage Example                                     | Description                                                                                               |
| ------------- | -------------------- | ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| BCELoss       | N/A                  | `--loss None`                                         | Binary Cross-Entropy Loss for binary classification tasks.                                                |
| FocalLoss     | alpha, gamma         | `--loss focal --alpha 0.8 --gamma 2`                  | Focal Loss to address class imbalance by focusing more on hard to classify examples.                      |
| DiceLoss      | smooth               | `--loss dice --smooth 1e-6`                           | Dice Loss for segmentation tasks, measuring overlap between predicted and actual segmentation maps.       |
| TverskyLoss   | alpha, beta, smooth  | `--loss tversky --alpha 0.5 --beta 0.5 --smooth 1e-6` | Tversky Loss allows flexibility by controlling the importance of false positives and false negatives.     |
| JaccardLoss   | smooth               | `--loss jaccard --smooth 1e-6`                        | Jaccard Loss (IoU Loss) for evaluating the similarity between the predicted and actual segmentation maps. |
| ComboLoss     | smooth, alpha, gamma | `--loss combo --smooth 1e-6 --alpha 0.5 --gamma 0.5`  | A combination of BCE, Focal, and Dice Losses to leverage their benefits for handling class imbalance.     |

### Usages

Below is a table that outlines various CLI command examples for training and testing models with and without the Attention UNet architecture, specifying different devices and whether or not a loss function is explicitly mentioned:

| Scenario                            | Command                                                                                                                        | Description                                                                                                                                   |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Training Without Attention UNet** | `python cli.py --train --image_path "./data/images.zip" --batch_size 32 --epochs 100 --lr 0.001 --device cuda`                 | Trains a standard UNet model on the CUDA device with specified parameters.                                                                    |
| **Training With Attention UNet**    | `python cli.py --train --image_path "./data/images.zip" --batch_size 32 --epochs 100 --lr 0.001 --attentionUNet --device cuda` | Trains an Attention UNet model on the CUDA device with specified parameters.                                                                  |
| **Training Without Specified Loss** | `python cli.py --train --image_path "./data/images.zip" --batch_size 32 --epochs 100 --lr 0.001 --device cuda`                 | Trains a model (default UNet) on the CUDA device without explicitly specifying the loss function. Defaults may apply based on implementation. |
| **Training With Specified Loss**    | `python cli.py --train --image_path "./data/images.zip" --batch_size 32 --epochs 100 --lr 0.001 --loss dice --device cuda`     | Trains a model with the specified Dice loss on the CUDA device.                                                                               |
| **Testing on CUDA**                 | `python cli.py --test --device cuda`                                                                                           | Tests a model on the CUDA device. Assumes model path or other necessary parameters are set by default or in the script.                       |
| **Training on MPS (Apple Silicon)** | `python cli.py --train --image_path "./data/images.zip" --batch_size 32 --epochs 100 --lr 0.001 --device mps`                  | Trains a standard UNet model on the MPS device with specified parameters. Useful for Apple Silicon users.                                     |
| **Training on CPU**                 | `python cli.py --train --image_path "./data/images.zip" --batch_size 32 --epochs 100 --lr 0.001 --device cpu`                  | Trains a standard UNet model on the CPU with specified parameters. Ideal for environments without dedicated GPU resources.                    |
| **Testing on MPS (Apple Silicon)**  | `python cli.py --test --device mps`                                                                                            | Tests a model on the MPS device. Useful for Apple Silicon users.                                                                              |
| **Testing on CPU**                  | `python cli.py --test --device cpu`                                                                                            | Tests a model on the CPU. Ideal for environments without dedicated GPU resources.                                                             |

This table provides a quick reference for users to understand how to utilize the CLI utility for different training and testing scenarios across various devices and configurations, including with or without specific loss functions.
