### Loader Module Usage

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
