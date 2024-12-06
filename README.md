# Continual Federated Learning

This repository implements **Continual Federated Learning** using **PyTorch**. The framework supports two main paradigms of continual learning:

1. **Task-Incremental Learning (Task-IL)**: Models are trained incrementally across different tasks, where each task has a separate output head.
2. **Class-Incremental Learning (Class-IL)**: Models learn incrementally while adapting to new classes in a unified output space.

## Features
- Fully modular and scalable architecture.
- Implemented using **PyTorch**.
- Supports easy execution with `.sh` scripts provided in the corresponding directories.

## How to Run
To execute experiments, navigate to the relevant directory (e.g., `task_incremental` or `class_incremental`) and run the provided `.sh` files:

```bash
bash run_experiment.sh
