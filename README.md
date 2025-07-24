# Waste Classification with PyTorch

A modular, reproducible pipeline for training and evaluating deep learning models for waste classification using PyTorch. This project uses Hydra for configuration, supports modern training practices, and is designed for clarity and extensibility.

---

## Project Structure

```
.
├── base/               # Abstract base classes for models, metrics, and trainers
├── configs/            # Hydra YAML configs for models, metrics, dataloaders, etc.
├── data/               # Dataset storage (not included in repo)
├── data_loader/        # Custom data loading and augmentation logic
├── logger/             # Logging and TensorBoard writer utilities
├── loss/               # Loss function implementations
├── metrics/            # Metric implementations (Accuracy, F1, Precision, Recall)
├── model/              # Model architectures (e.g., EfficientNet)
├── outputs/            # Output directory for logs, checkpoints, predictions
├── trainer/            # Training and inference logic
├── utils/              # Utility functions (seed setting, directory management, etc.)
│
├── train.py            # Main training script
├── test.py             # Main evaluation/inference script
├── confusion_matrix.py # Script to visualize confusion matrix from predictions
│
├── Dockerfile          # Docker support for reproducible training environments
├── pyproject.toml      # Python project and dependency management
└── README.md           # Project documentation
```

---

## Getting Started

### 1. Install Dependencies

We recommend using [uv](https://github.com/astral-sh/uv):

```sh
uv venv --python 3.13
uv sync
```

### 2. Prepare the Dataset

- Place your dataset in `data/waste_dataset/` following the [ImageFolder](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html) structure:
  ```
  data/waste_dataset/
      class_0/
          img1.jpg
          ...
      class_1/
          ...
      ...
  ```

### 3. Configure Your Experiment

- Edit YAML files in [`configs/`](configs/) to change model, optimizer, metrics, data loader, etc.
- Examples:
  - [`configs/train.yaml`](configs/train.yaml)
  - [`configs/model/waste_classification.yaml`](configs/model/waste_classification.yaml)
  - [`configs/metrics/waste_classification.yaml`](configs/metrics/waste_classification.yaml)

### 4. Train the Model

```sh
python train.py
```
- Checkpoints and logs will be saved in `outputs/` or as configured.

### 5. Evaluate / Inference

```sh
python test.py
```
- Predictions will be saved to `prediction.csv` (configurable in `configs/test.yaml`).

### 6. Visualize Results

- To plot a confusion matrix from predictions:
  ```sh
  python confusion_matrix.py
  ```
- For TensorBoard logs:
  ```sh
  tensorboard --logdir outputs/
  ```

---

## Configuration

- All experiment settings are controlled via YAML files in [`configs/`](configs/).
- Use [Hydra](https://hydra.cc/) to override configs from the command line if needed.

---

## Extending

- **Add a new model**: Implement in [`model/`](model/), register in [`configs/model/`](configs/model/).
- **Add a new metric**: Implement in [`metrics/`](metrics/), register in [`configs/metrics/`](configs/metrics/).
- **Custom data loading**: Modify or extend [`data_loader/`](data_loader/).
