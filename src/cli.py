import argparse
from dataloader import Loader
from UNet import UNet
from AttentionUNet import AttentionUNet
from trainer import Trainer
from test import Charts


def cli():
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
        charts = Charts(device="mps")
        charts.test()


if __name__ == "__main__":
    cli()
