import sys
import os
import torch
import torch.nn as nn
import joblib

sys.path.append("src/")

from config import (
    RAW_PATH,
    PROCESSED_PATH,
    TRAIN_CHECKPOINT_PATH,
    BEST_MODEL_PATH,
    TRAIN_IMAGES_PATH,
    TEST_IMAGE_PATH,
)


def dump(value, filename):
    """
    Serialize and save a Python object to a file using joblib.

    This function serializes a given value and saves it to a specified filename. It is
    useful for persisting models, data, or other serializable Python objects to disk.

    Parameters:
    - value: The Python object to serialize. This object must be serializable by joblib.
    - filename: The path to the file where the serialized object will be saved. Must include
      the file name and extension.

    Raises:
    - Exception: If either `value` or `filename` is None, indicating that the required
      information for dumping is missing.
    """
    if value is not None and filename is not None:
        joblib.dump(value=value, filename=filename)
    else:
        raise Exception("Value or filename cannot be None".capitalize())


def load(filename):
    """
    Load and deserialize a Python object from a file using joblib.

    This function reads from a specified filename and deserializes the contained Python
    object. It is typically used to load models, data, or other Python objects that have
    been previously serialized with joblib.

    Parameters:
    - filename: The path to the file from which the object will be deserialized. Must
      include the file name and extension.

    Returns:
    - The deserialized Python object.

    Raises:
    - Exception: If `filename` is None, indicating that no file name was provided.
    """
    if filename is not None:
        return joblib.load(filename)
    else:
        raise Exception("Filename cannot be None".capitalize())


def device_init(device="mps"):
    """
    Initialize and return a PyTorch device based on availability and user preference.

    This function facilitates the use of different computing devices (e.g., MPS, CUDA) with
    PyTorch. It checks for the availability of the specified device and falls back to CPU if
    the preferred device is not available.

    Parameters:
    - device: A string indicating the preferred device. Options include "mps" for Apple Metal
      Performance Shaders, "cuda" for NVIDIA CUDA, and any other string defaults to CPU. The
      default is "mps".

    Returns:
    - A torch.device object representing the initialized computing device.
    """
    if device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device("cpu")


def weight_init(m):
    """
    Initialize the weights of a PyTorch module according to its type.

    This function applies specific initialization schemes to the weights of Convolutional
    and Batch Normalization layers within a PyTorch module. It uses He initialization
    (also known as Kaiming initialization) for Convolutional layers and normal
    initialization for Batch Normalization layers.

    Parameters:
    - m: A PyTorch module whose weights will be initialized.

    Note:
    - This function is intended to be passed as an argument to the `apply` method of
      a PyTorch model for recursive application to each module.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


def clean():
    """
    Clean up the project directories by removing certain files and folders.

    This function removes various files and folders to clean up the project directories.
    It removes the following:
    - All files in the 'segmented' folder within the RAW_PATH directory.
    - All files in the PROCESSED_PATH directory.
    - All files in the TRAIN_CHECKPOINT_PATH directory.
    - All files in the BEST_MODEL_PATH directory.
    - All files in the TRAIN_IMAGES_PATH directory.
    - All files in the TEST_IMAGE_PATH directory.

    Raises:
        Exception: If any of the specified directories do not exist.
    """
    if os.path.exists(RAW_PATH):
        image_folder = os.path.join(RAW_PATH, "segmented")
        os.system(f"rm -rf {image_folder}")
    else:
        raise Exception("Raw data folder does not exist".capitalize())

    if os.path.exists(PROCESSED_PATH):
        for file in os.listdir(PROCESSED_PATH):
            os.remove(os.path.join(PROCESSED_PATH, file))
    else:
        raise Exception("Processed data folder does not exist".capitalize())

    if os.path.exists(TRAIN_CHECKPOINT_PATH):
        for file in os.listdir(TRAIN_CHECKPOINT_PATH):
            os.remove(os.path.join(TRAIN_CHECKPOINT_PATH, file))
    else:
        raise Exception("Training checkpoint folder does not exist".capitalize())

    if os.path.exists(BEST_MODEL_PATH):
        for file in os.listdir(BEST_MODEL_PATH):
            os.remove(os.path.join(BEST_MODEL_PATH, file))
    else:
        raise Exception("Training checkpoint folder does not exist".capitalize())

    if os.path.exists(TRAIN_IMAGES_PATH):
        for file in os.listdir(TRAIN_IMAGES_PATH):
            os.remove(os.path.join(TRAIN_IMAGES_PATH, file))
    else:
        raise Exception("Train images folder does not exist".capitalize())

    if os.path.exists(TEST_IMAGE_PATH):
        for file in os.listdir(TEST_IMAGE_PATH):
            os.remove(os.path.join(TEST_IMAGE_PATH, file))
    else:
        raise Exception("Test images folder does not exist".capitalize())
