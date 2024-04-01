import argparse
from dataloader import Loader
from trainer import Trainer
from test import Charts


def cli():
    """
    Command Line Interface for configuring and running training or testing processes
    for UNet and Attention UNet models.

    This function parses command-line arguments to configure data loading, model training,
    and testing. It supports a range of options allowing for specification of dataset
    properties (such as path, batch size, split ratio, and image size), training
    parameters (like epochs, learning rate, loss function, and regularization methods),
    and model choices (UNet or Attention UNet). Depending on the arguments, it
    initializes the data loading process, followed by model training or testing.

    Arguments:
        --image_path (str): Path to the zip file containing the images.
        --batch_size (int): Batch size for the dataloader.
        --split_ratio (float): Split ratio for training and validation datasets.
        --image_size (int): The height and width of images after resizing.
        --epochs (int): Number of epochs to train.
        --lr (float): Learning rate for the optimizer.
        --loss (str): Loss function to use.
        --attentionUNet (bool): Whether to use Attention UNet instead of standard UNet.
        --display (bool): Display training progress and metrics.
        --device (str): Computation device to use ('cuda', 'mps', etc.).
        --smooth (float): Smooth value for certain regularization.
        --alpha (float): Alpha value for specific loss functions or regularization.
        --gamma (float): Gamma value for specific loss functions.
        --is_l1 (bool): Enable L1 regularization.
        --is_l2 (bool): Enable L2 regularization.
        --is_elastic (bool): Apply elastic transformation to the data.
        --is_weight_clip (bool): Enable weight clipping.
        --train (flag): Flag to initiate the training process.
        --test (flag): Flag to initiate the testing process.

    Based on the provided arguments, the function will orchestrate the setup and execution
    of data loading, model training, or model testing. For training, it initializes a
    data loader, processes the dataset, and begins training with specified configurations.
    For testing, it initializes the test environment and runs model evaluation.

    Example Command:
        python cli.py --train --image_path "./data/images.zip" --batch_size 32 --epochs 100
                       --lr 0.001 --attentionUNet --device cuda

    This would train an Attention UNet model on the data located at "./data/images.zip",
    with a batch size of 32 for 100 epochs, using a learning rate of 0.001 on the CUDA device.
    """
    parser = argparse.ArgumentParser(
        description="CLI - Command LIne Interface for UNet".title()
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
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs".capitalize()
    )
    parser.add_argument(
        "--lr", type=float, default=1e-2, help="Learning rate".capitalize()
    )
    parser.add_argument(
        "--loss", type=str, default=None, help="Loss function".capitalize()
    )
    parser.add_argument(
        "--attentionUNet", type=bool, default=False, help="Attention UNet".capitalize()
    )
    parser.add_argument(
        "--is_attentionUNet",
        type=bool,
        default=False,
        help="Attention UNet".capitalize(),
    )
    parser.add_argument(
        "--display", type=bool, default=True, help="Display progress".capitalize()
    )
    parser.add_argument("--device", type=str, default="mps", help="Device".capitalize())
    parser.add_argument(
        "--smooth", type=float, default=0.01, help="Smooth value".capitalize()
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Alpha value".capitalize()
    )
    parser.add_argument(
        "--gamma", type=float, default=2, help="Gamma value".capitalize()
    )
    parser.add_argument(
        "--is_l1", type=bool, default=False, help="L1 value".capitalize()
    )
    parser.add_argument(
        "--is_l2", type=bool, default=False, help="L2 value".capitalize()
    )
    parser.add_argument(
        "--is_elastic", type=bool, default=False, help="Elastic Transform".capitalize()
    )
    parser.add_argument(
        "--is_weight_clip", type=bool, default=False, help="Weight Clip".capitalize()
    )
    parser.add_argument("--train", action="store_true", help="Train model".capitalize())
    parser.add_argument("--test", action="store_true", help="Test model".capitalize())

    args = parser.parse_args()

    if args.train:

        loader = Loader(
            image_path=args.image_path,
            batch_size=args.batch_size,
            split_ratio=args.split_ratio,
            image_size=args.image_size,
        )

        loader.unzip_folder()

        _ = loader.create_dataloader()

        Loader.details_dataset()

        trainer = Trainer(
            epochs=args.epochs,
            lr=args.lr,
            loss=args.loss,
            is_attentionUNet=args.attentionUNet,
            is_l1=args.is_l1,
            is_l2=args.is_l2,
            is_elastic=args.is_elastic,
            is_weight_clip=args.is_weight_clip,
            alpha=args.alpha,
            gamma=args.gamma,
            display=args.display,
            device=args.device,
            smooth=args.smooth,
        )

        trainer.train()

    elif args.test:
        charts = Charts(device=args.device, is_attentionUNet=args.is_attentionUNet)
        charts.test()


if __name__ == "__main__":
    cli()
