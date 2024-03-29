import sys
import os
import logging
import argparse
import zipfile
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="./logs/dataloader.log",
)

from utils import dump, load
from config import RAW_PATH, PROCESSED_PATH


class Loader(Dataset):
    def __init__(self, image_path=None, image_size=128, batch_size=4, split_ratio=0.30):
        self.image_path = image_path
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.image_size = image_size
        self.channels = 1
        self.images = list()
        self.masks = list()

    def unzip_folder(self):
        if os.path.exists(RAW_PATH):
            with zipfile.ZipFile(self.image_path, "r") as zip_ref:
                zip_ref.extractall(os.path.join(RAW_PATH, "segmented"))
        else:
            raise Exception("Raw data folder does not exist".capitalize())

    def base_transformation(self):
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def mask_transformation(self):
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.Grayscale(num_output_channels=self.channels),
            ]
        )

    def split_dataset(self, **kwargs):
        images = kwargs["images"]
        masks = kwargs["masks"]

        return train_test_split(
            images, masks, test_size=self.split_ratio, random_state=42
        )

    def create_dataloader(self):
        images = os.listdir(os.path.join(RAW_PATH, "segmented"))[0]
        masks = os.listdir(os.path.join(RAW_PATH, "segmented"))[1]

        try:
            images = os.path.join(RAW_PATH, "segmented", images)
            masks = os.path.join(RAW_PATH, "segmented", masks)
        except Exception as e:
            print(e)

        for image in os.listdir(images):
            if image not in os.listdir(masks):
                continue
            else:
                image_path = os.path.join(images, image)
                mask_path = os.path.join(masks, image)

                self.images.append(
                    self.base_transformation()(Image.fromarray(cv2.imread(image_path)))
                )
                self.masks.append(
                    self.mask_transformation()(Image.fromarray(cv2.imread(mask_path)))
                )

        image_split = self.split_dataset(images=self.images, masks=self.masks)

        if os.path.exists(PROCESSED_PATH):

            dataloader = DataLoader(
                dataset=list(zip(self.images, self.masks)),
                batch_size=self.batch_size,
                shuffle=True,
            )

            train_dataloader = DataLoader(
                dataset=list(zip(image_split[0], image_split[2])),
                batch_size=self.batch_size,
                shuffle=True,
            )

            test_dataloader = DataLoader(
                dataset=list(zip(image_split[1], image_split[3])),
                batch_size=self.batch_size * 4,
                shuffle=True,
            )

            try:
                dump(
                    value=dataloader,
                    filename=os.path.join(PROCESSED_PATH, "dataloader.pkl"),
                )

                dump(
                    value=train_dataloader,
                    filename=os.path.join(PROCESSED_PATH, "train_dataloader.pkl"),
                )

                dump(
                    value=test_dataloader,
                    filename=os.path.join(PROCESSED_PATH, "test_dataloader.pkl"),
                )

            except Exception as e:
                print(e)
        else:
            raise Exception("Processed data folder does not exist".capitalize())

        return dataloader

    @staticmethod
    def details_dataset():
        if os.path.exists(PROCESSED_PATH):

            dataloader = load(os.path.join(PROCESSED_PATH, "dataloader.pkl"))
            images, masks = next(iter(dataloader))
            print(
                "Total number of the images in the dataset is {}".format(
                    sum(image.size(0) for image, _ in dataloader)
                )
            )
            print(
                "Total number of the masks in the dataset is {}\n\n".format(
                    sum(masks.size(0) for _, masks in dataloader)
                )
            )
            print(
                "The shape of the images is {}\nThe shape of the masks is {}".format(
                    images.size(), masks.size()
                )
            )

        else:
            raise Exception("Processed data folder does not exist".capitalize())

    @staticmethod
    def data_normalized(**kwargs):
        return (kwargs["data"] - kwargs["data"].min()) / (
            kwargs["data"].max() - kwargs["data"].min()
        )

    @staticmethod
    def display_images():
        if os.path.exists(PROCESSED_PATH):
            dataloader = load(os.path.join(PROCESSED_PATH, "test_dataloader.pkl"))
            images, masks = next(iter(dataloader))

            plt.figure(figsize=(30, 15))

            for index, image in enumerate(images):
                image = image.permute(1, 2, 0)
                mask = masks[index].permute(1, 2, 0)

                image = Loader.data_normalized(data=image)
                mask = Loader.data_normalized(data=mask)

                plt.subplot(2 * 4, 2 * 4, 2 * index + 1)
                plt.imshow(image)
                plt.title("Image")
                plt.axis("off")

                plt.subplot(2 * 4, 2 * 4, 2 * index + 2)
                plt.imshow(mask)
                plt.title("Mask")
                plt.axis("off")

            plt.tight_layout()
            plt.show()

        else:
            raise Exception("Processed data folder does not exist".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Define the dataloader for UNet".title()
    )

    parser.add_argument(
        "--image_path",
        type=str,
        help="Path to the zip file containing the images".capitalize(),
    )
    parser.add_argument(
        "--batch_size", type=int, help="Batch size for the dataloader".capitalize()
    )
    parser.add_argument(
        "--split_ratio", type=float, help="Split ratio for the dataset".capitalize()
    )
    parser.add_argument(
        "--image_size", type=int, help="Image size for the dataloader".capitalize()
    )

    args = parser.parse_args()

    if args.image_path:
        if args.batch_size and args.split_ratio and args.image_size:
            loader = Loader(
                image_path=args.image_path,
                batch_size=args.batch_size,
                split_ratio=args.split_ratio,
                image_size=args.image_size,
            )

            loader.unzip_folder()

            dataloader = loader.create_dataloader()

            logging.info(Loader.details_dataset())
            Loader.display_images()

        else:
            raise Exception("Missing arguments".capitalize())
    else:
        raise Exception("Missing arguments".capitalize())