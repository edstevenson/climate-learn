# Core Components

## Data module

- Implements data loading and processing for climate datasets
- two main types of data modules:
    - IterDataModule: For iterative/streaming data processing
    - MapDataModule: For memory-mapped data processing
- Supports multiple climate datasets (ERA5, CMIP6, PRISM, ClimateBench)

## Models module
Supports multiple model types:
- Vision Transformers (ViT)
- ResNet
- UNet
- Baseline models (interpolation, climatology)
- Core model implementation in LitModule which extends PyTorch Lightning

## Tasks
- The package supports three main climate modeling tasks:
    - Forecasting (direct, iterative, and continuous)
    - Downscaling (spatial resolution enhancement)
    - Climate projection (ClimateBench tasks)

## Metrics and Evaluation
- Custom climate-specific metrics
- Support for climatology-based evaluation
- Implements various loss functions and transformations

# Key files structure
```
climate-learn/
├── src/climate_learn/
│   ├── data/                 # Data loading and processing
│   │   ├── itermodule.py    # Iterative data loading
│   │   ├── mapmodule.py     # Memory-mapped data loading
│   │   └── climatebench_module.py
│   ├── models/              # Model implementations
│   │   ├── module.py        # Base Lightning module
│   │   └── hub/            # Model architectures
│   ├── metrics/             # Evaluation metrics
|   ├── transforms/         # ?
│   └── utils/               # Helper functions
├── experiments/             # Training scripts
│   ├── forecasting/
│   ├── downscaling/
│   └── climate_projection/
```