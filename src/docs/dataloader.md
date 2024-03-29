# Image Segmentation DataLoader

The `Loader` class is designed to facilitate the loading, preprocessing, splitting, and normalization of image and mask datasets for neural network training, particularly tailored for image segmentation tasks like those involving U-Net architectures.

## Features

- **Data Extraction:** Unzips image datasets for processing.
- **Image Preprocessing:** Applies transformations to both images and masks, including resizing, normalization, and converting masks to grayscale.
- **Dataset Splitting:** Easily split datasets into training and testing sets.
- **Data Normalization:** Normalizes image data to the range [0, 1].
- **Data Loading:** Creates DataLoader objects for efficient batch processing during neural network training.

## Parameters & Methods

| Parameter   | Type  | Description                                        |
| ----------- | ----- | -------------------------------------------------- |
| image_path  | str   | Path to the zip file containing the images.        |
| image_size  | int   | Target size to which the images should be resized. |
| batch_size  | int   | Number of images and masks in each batch of data.  |
| split_ratio | float | Ratio of the dataset to be used as test set.       |

**Methods:**

- `unzip_folder()`: Extracts images from a zip file to a specified directory.
- `base_transformation()`: Applies basic transformations to images.
- `mask_transformation()`: Applies transformations to mask images.
- `split_dataset(images, masks)`: Splits the dataset into training and testing sets.
- `create_dataloader()`: Prepares DataLoader objects for training and testing datasets.
- `details_dataset()`: Prints details about the dataset.
- `data_normalized(data)`: Normalizes a given dataset.
- `display_images()`: Displays images and their corresponding masks from the dataset.

**Notes:**

- Ensure RAW_PATH and PROCESSED_PATH are correctly configured in your config file.
- This class requires the 'torch', 'torchvision', 'PIL', and 'cv2' libraries.
  """

### Initializing the Loader

To start using the Loader, initialize it with the path to your dataset's zip file, desired image size, batch size, and split ratio:

```python
from loader import Loader

loader = Loader(image_path="path/to/images.zip", image_size=128, batch_size=4, split_ratio=0.3)
```

### Extracting and Processing Data

First, unzip the dataset and then create a DataLoader object:

```python
loader.unzip_folder()
dataloader = loader.create_dataloader()
```

### Displaying Dataset Details and Images

To print details about the dataset and visualize some images along with their masks:

```python
Loader.details_dataset()
Loader.display_images()
```

## Script Usage

The functionality can also be accessed via a command-line script. To use it, execute the following command with the necessary arguments:

```bash
python dataloader_script.py --image_path "/path/to/images.zip" --batch_size 4 --split_ratio 0.3 --image_size 128
```
