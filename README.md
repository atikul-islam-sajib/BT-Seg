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

![AC-GAN - Medical Image Dataset Generator with class labels: Gif file](https://github.com/atikul-islam-sajib/BT-Seg/blob/main/research/files/attentionUNet-output/train_masks.gif)

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

#### Training the Model

To train the model, you need a dataset in a zip file specified by `--image_path`, along with any other configurations you wish to customize.

- **Using CUDA (for NVIDIA GPUs):**

```
python cli.py --image_path "/path/to/dataset.zip" --batch_size 4 --image_size 128 --split_ratio 0.25 --epochs 50 --lr 0.001 --loss dice --display True --smooth_value 0.01 --alpha 0.25 --gamma 2 --device cuda --train
```

- **Using MPS (for Apple Silicon GPUs):**

```
python cli.py --image_path "/path/to/dataset.zip" --batch_size 4 --image_size 128 --split_ratio 0.25 --epochs 50 --lr 0.001 --loss dice --display True --smooth_value 0.01 --alpha 0.25 --gamma 2  --device mps --train
```

- **Using CPU:**

```
python cli.py --image_path "/path/to/dataset.zip" --batch_size 4 --image_size 128 --split_ratio 0.25 --epochs 50 --lr 0.001 --loss dice --display True --smooth_value 0.01 --alpha 0.25 --gamma 2 --device cpu --train
```

#### Testing the Model

Ensure you specify the device using `--device` if different from the default. The test process can be initiated with the `--test` flag.

- **Using CUDA (for NVIDIA GPUs):**

```
python cli.py --device cuda --test
```

- **Using MPS (for Apple Silicon GPUs):**

```
python cli.py --device mps --test
```

- **Using CPU:**

```
python cli.py --device cpu --test
```

#### Import Custom Modules

First, ensure that you have the necessary modules available in your Python environment. These modules include functionalities for data loading, model definition, training, and evaluation.

```python
from src.dataloader import Loader
from src.UNet import UNet
from src.trainer import Trainer
from src.test import Charts
```

## DataLoader

The `Loader` class is responsible for preparing the dataset. It unzips the dataset, splits it into training and testing sets based on the provided ratio, and creates DataLoaders for both.

To use the DataLoader, ensure you have your dataset in a zip file. Specify the path to this file along with other parameters such as batch size, split ratio, and image size.

Example:

```python
from src.dataloader import Loader

loader = Loader(
    image_path="path/to/your/dataset.zip",
    batch_size=4,
    split_ratio=0.25,
    image_size=128
)
loader.unzip_folder()
loader.create_dataloader()
```

### Loss Functions

The training process supports several loss functions, allowing you to choose the one that best fits your project's needs. Below is a table describing the available loss functions and how to specify each in the training command or configuration.

| Loss Function | Call              | Description                                                                                                                                                      |
| ------------- | ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dice          | `loss="dice"`     | Measures the overlap between the predicted segmentation and the ground truth. Ideal for binary segmentation tasks.                                               |
| Jaccard       | `loss="jaccard"`  | Also known as the Intersection over Union (IoU) loss. Similar to Dice but with a different formula. Good for evaluating the accuracy of object detection models. |
| IoU           | `loss="IoU"`      | Another name for Jaccard loss.                                                                                                                                   |
| Combo         | `loss="combo"`    | Combines Dice and a cross-entropy loss to leverage the benefits of both. Useful for unbalanced datasets.                                                         |
| Focal         | `loss="focal"`    | Focuses on hard-to-classify examples by reducing the relative loss for well-classified examples. Useful for datasets with imbalanced classes.                    |
| Dice_BCE      | `loss="dice_bce"` | A combination of Dice and Binary Cross-Entropy (BCE) losses. Offers a balance between shape similarity and pixel-wise accuracy.                                  |
| None          | `loss=None`       | IT will trigger the BCELoss                                                                                                                                      |

### Trainer

The `Trainer` class manages the training process, including setting up the loss function, optimizer, and device (CPU, CUDA, MPS). It also handles the training epochs and displays progress if enabled.

To train your model, configure the `Trainer` with your desired settings.

Example:

```python
from src.trainer import Trainer

trainer = Trainer(
    epochs=100,
    lr=0.01,
    loss="dice", # can be "jaccard", "IoU", "combo", "focal", "dice_bce", "None"
    alpha=0.5,
    gamma=2,
    display=True,
    device="cuda",  # Can be "cpu", "cuda", or "mps"
    smooth_value=0.01
)
trainer.train()
```

### Charts

After training, you can test and visualize the model's performance using the `Charts` class. This class allows you to evaluate the trained model on the test dataset and generate performance metrics.

Example:

```python
from src.test import Charts

charts = Charts(device="mps")  # Specify the device used for testing
charts.test()
```

#### 5. Visualize Results

Visualize the test results and the loss curves by displaying the generated images. Ensure you specify the correct paths to the images.

```python
from IPython.display import Image

# Display the result image
Image("/content/U-Net/outputs/test_images/result.png")

# Display the loss curve image
Image("/content/U-Net/outputs/test_images/loss.png")
```

## Contributing

Contributions to improve this implementation of U-Net are welcome. Please follow the standard fork-branch-pull request workflow.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
