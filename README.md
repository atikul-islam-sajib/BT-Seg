# U-Net & AttentionUNet for Semantic Image Segmentation

U-Net is a convolutional neural network designed for semantic image segmentation. This implementation of U-Net is tailored for high performance on various image segmentation tasks, allowing for precise object localization within images.

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20220614121231/Group14.jpg" alt="AC-GAN - Medical Image Dataset Generator: Generated Image with labels">

#### UNet - output

<img src="https://github.com/atikul-islam-sajib/BT-Seg/blob/main/research/files/Unet-output/result.jpg">

#### AttentionUNet - input

<img src="https://github.com/atikul-islam-sajib/BT-Seg/blob/main/research/files/attentionUNet-output/result.jpg">

## Features

| Feature                          | Description                                                                                                                                                                                                           |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Efficient Implementation**     | Utilizes an optimized U-Net model architecture for superior performance on diverse image segmentation tasks.                                                                                                          |
| **Custom Dataset Support**       | Features easy-to-use data loading utilities that seamlessly accommodate custom datasets, requiring minimal configuration.                                                                                             |
| **Training and Testing Scripts** | Provides streamlined scripts for both training and testing phases, simplifying the end-to-end workflow.                                                                                                               |
| **Visualization Tools**          | Equipped with tools for tracking training progress and visualizing segmentation outcomes, enabling clear insight into model effectiveness.                                                                            |
| **Custom Training via CLI**      | Offers a versatile command-line interface for personalized training configurations, enhancing flexibility in model training.                                                                                          |
| **Import Modules**               | Supports straightforward integration into various projects or workflows with well-documented Python modules, simplifying the adoption of U-Net, AttentionUNet functionality.                                          |
| **Multi-Platform Support**       | Guarantees compatibility with various computational backends, including MPS for GPU acceleration on Apple devices, CPU, and CUDA for Nvidia GPU acceleration, ensuring adaptability across different hardware setups. |

## Demo - During training

<table>
  <tr>
    <td><img src="https://github.com/atikul-islam-sajib/BT-Seg/blob/main/research/files/attentionUNet-output/train_masks.gif" alt="UNet" style="width: 100%; max-width: 400px;"/></td>
    <td><img src="https://github.com/atikul-islam-sajib/BT-Seg/blob/main/research/files/attentionUNet-output/train_masks.gif" alt="AttentionUNet" style="width: 100%; max-width: 400px;"/></td>
  </tr>
  <tr>
    <td align="center">UNet</td>
    <td align="center">AttentionUNet</td>
  </tr>
</table>

## Getting Started

## Requirements

| Requirement             | Description                                                                                                     |
| ----------------------- | --------------------------------------------------------------------------------------------------------------- |
| **Python Version**      | Python 3.9 or newer is required for compatibility with the latest features and library support.                 |
| **CUDA-compatible GPU** | Access to a CUDA-compatible GPU is recommended for training and testing with CUDA acceleration.                 |
| **Python Libraries**    | Essential libraries include: **torch**, **matplotlib**, **numpy**, **PIL**, **scikit-learn**, **opencv-python** |

## Installation Instructions

Follow these steps to get the project set up on your local machine:

| Step | Instruction                                  | Command                                                        |
| ---- | -------------------------------------------- | -------------------------------------------------------------- |
| 1    | Clone this repository to your local machine. | **git clone https://github.com/atikul-islam-sajib/BT-Seg.git** |
| 2    | Navigate into the project directory.         | **cd BT-Seg**                                                  |
| 3    | Install the required Python packages.        | **pip install -r requirements.txt**                            |

## Project Structure

This project is thoughtfully organized to support the development, training, and evaluation of the U-Net model efficiently. Below is a concise overview of the directory structure and their specific roles:

- **checkpoints/**
  - Stores model checkpoints during training for later resumption.
- **best_model/**

  - Contains the best-performing model checkpoints as determined by validation metrics.

- **train_models/**

  - Houses all model checkpoints generated throughout the training process.

- **data/**

  - **processed/**: Processed data ready for modeling, having undergone normalization, augmentation, or encoding.
  - **raw/**: Original, unmodified data serving as the baseline for all preprocessing.

- **logs/**

  - **Log** files for debugging and tracking model training progress.

- **metrics/**

  - Files related to model performance metrics for evaluation purposes.

- **outputs/**

  - **test_images/**: Images generated during the testing phase, including segmentation outputs.
  - **train_gif/**: GIFs compiled from training images showcasing the model's learning progress.
  - **train_images/**: Images generated during training for performance visualization.

- **research/**

  - **notebooks/**: Jupyter notebooks for research, experiments, and exploratory analyses conducted during the project.

- **src/**

  - Source code directory containing all custom modules, scripts, and utility functions for the U-Net model.

- **unittest/**
  - Unit tests ensuring code reliability, correctness, and functionality across various project components.

### Dataset Organization for Semantic Image Segmentation

The dataset is organized into three categories for semantic image segmentation tasks: benign, normal, and malignant. Each category directly contains paired images and their corresponding segmentation masks, stored together to simplify the association between images and masks.

## Directory Structure:

```
segmentation/
├── images/
│ ├── 1.png
│ ├── 2.png
│ ├── ...
│ ├── ...
├── masks/
│ ├── 1.png
│ ├── 2.png
│ ├── ...
│ ├── ...
```

#### Naming Convention:

- **Images and Masks**: Within each category folder, images and their corresponding masks are stored together. The naming convention for images is `(n).png`, and for masks, it is in Segmented `(n).png`, where n represents the type of the image (benign, normal, or malignant), and `(n)` is a unique identifier. This convention facilitates easy identification and association of each image with its respective mask.

For detailed documentation on the dataset visit the [Dataset - Kaggle](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation).

### Documentation U-Net

For detailed documentation on the implementation and usage, visit the -> [U-Net Documentation](https://atikul-islam-sajib.github.io/U-Net-MultiClass-Deploy/).

### User's guidance Notebook for BT-Seg

For detailed implementation and usage - CLI, visit the -> [U-Net: CLI Notebook](./research/notebooks/ModelTrain-CLI.ipynb).

For detailed implementation and usage - Custom Modules, visit the -> [U-Net: Custom Modules Notebook](./research/notebooks/ModelTrain-Modules.ipynb).

# Command Line Usage

```
python cli.py --help
```

### CLI - Arguments

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

### Supported Loss Functions

The CLI tool supports various loss functions, each with specific parameters for fine-tuning the training process.

| Loss Function | Parameters           | CLI Usage Example                                     | Description                                                                                               |
| ------------- | -------------------- | ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| BCELoss       | N/A                  | `--loss None`                                         | Binary Cross-Entropy Loss for binary classification tasks.                                                |
| FocalLoss     | alpha, gamma         | `--loss focal --alpha 0.8 --gamma 2`                  | Focal Loss to address class imbalance by focusing more on hard to classify examples.                      |
| DiceLoss      | smooth               | `--loss dice --smooth 1e-6`                           | Dice Loss for segmentation tasks, measuring overlap between predicted and actual segmentation maps.       |
| TverskyLoss   | alpha, beta, smooth  | `--loss tversky --alpha 0.5 --beta 0.5 --smooth 1e-6` | Tversky Loss allows flexibility by controlling the importance of false positives and false negatives.     |
| JaccardLoss   | smooth               | `--loss jaccard --smooth 1e-6`                        | Jaccard Loss (IoU Loss) for evaluating the similarity between the predicted and actual segmentation maps. |
| ComboLoss     | smooth, alpha, gamma | `--loss combo --smooth 1e-6 --alpha 0.5 --gamma 0.5`  | A combination of BCE, Focal, and Dice Losses to leverage their benefits for handling class imbalance.     |

### Training and Testing

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

### Training and Testing with Custom Modules

#### Loader Module Usage

The `Loader` module is designed for data preparation tasks such as loading, splitting, and optionally applying transformations.

```python
# Initialize the Loader with dataset configurations
loader = Loader(
    image_path="path/to/your/images.zip",
    batch_size=32,
    split_ratio=0.8,
    image_size=256,
)

# Unzip the dataset if necessary
loader.unzip_folder()

# Create a PyTorch DataLoader
dataloader = loader.create_dataloader()

# Display dataset details (optional)
Loader.details_dataset()

# Display a batch of images (optional)
Loader.display_images()
```

### Trainer Module Usage

The `Trainer` module handles the training process, accepting various parameters to configure the training session.

```python
# Initialize the Trainer with training configurations
trainer = Trainer(
    epochs=100,
    lr=0.001,
    loss="Dice",
    is_attentionUNet=False,  # Set True to use Attention UNet
    is_l1=False,
    is_l2=False,
    is_elastic=False,
    is_weight_clip=False,
    alpha=0.5,
    gamma=2.0,
    display=True,
    device="cuda",  # Or "mps", "cpu"
    smooth=1e-6,
)

# Start the training process
trainer.train()
```

### Charts Module Usage

The `Charts` module is utilized for evaluating the model's performance post-training and generating relevant charts or metrics.

```python
# Initialize the Charts for performance evaluation
charts = Charts(device="cuda", is_attentionUNet=False)  # Set True if testing Attention UNet

# Execute the testing and generate charts
charts.test()
```

#### 5. Visualize Results

Visualize the test results and the loss curves by displaying the generated images. Ensure you specify the correct paths to the images.

```python
from IPython.display import Image

# Display the result image
Image("/content/BT-Seg/outputs/test_images/result.png")

# Display the loss curve image
Image("/content/BT-Seg/outputs/test_images/loss.png")
```

## Contributing

Contributions to improve this implementation of U-Net are welcome. Please follow the standard fork-branch-pull request workflow.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
