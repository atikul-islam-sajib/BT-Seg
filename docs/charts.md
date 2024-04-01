# Charts Project

The Charts project is designed to visualize the performance of segmentation models, specifically UNet and AttentionUNet, through various charts and GIFs showcasing training progress. This utility is intended to aid in the analysis and presentation of model effectiveness in tasks such as medical image segmentation.

## Usage

The main functionality is encapsulated in the `Charts` class, which can be run from the command line. It supports various operations, including model testing, data plotting, and GIF generation to visualize training progress.

### Command Line Arguments

| Argument   | Type   | Default | Description                                          |
| ---------- | ------ | ------- | ---------------------------------------------------- |
| `--device` | string | `"mps"` | Specifies the computing device for model operations. |

### Class `Charts`

#### Initialization

| Parameter          | Type   | Default | Description                                |
| ------------------ | ------ | ------- | ------------------------------------------ |
| `device`           | string | `"mps"` | The computing device for model operations. |
| `is_attentionUNet` | bool   | `None`  | Determines if AttentionUNet model is used. |

#### Methods

| Method                            | Description                                               |
| --------------------------------- | --------------------------------------------------------- |
| `__init__(self, device)`          | Initializes the Charts object with the specified device.  |
| `select_best_model(self)`         | Selects the best model based on saved checkpoints.        |
| `obtain_dataloader(self)`         | Loads the test dataset.                                   |
| `data_normalized(self, **kwargs)` | Normalizes the data for plotting.                         |
| `plot_data_comparison(self)`      | Plots a comparison of images, masks, and predicted masks. |
| `generate_gif(self)`              | Generates a GIF from training images.                     |
| `test(self)`                      | Executes the test phase, including visualization.         |

## Examples

To test the model and generate visualizations, run the following command:

```bash
python charts.py --device cuda
```

This command will run the `test` method on a CUDA device if available, evaluating the model and producing visualizations of its performance.

## Notes

- Ensure that the paths in the `config.py` file correctly point to your model, dataset, and other relevant files.
- The `device` argument allows you to specify the hardware for computations. It defaults to "mps" (Apple Metal Performance Shaders), but can be set to "cuda" for NVIDIA GPUs or "cpu" for the system's central processing unit.
- The project is configured to work with MKDocs for easy documentation. To serve the documentation locally, install MKDocs and run `mkdocs serve` in the project's root directory.
