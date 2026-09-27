# MLOps Assignment 2

## Setup

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

## Data & Model Versioning

This project uses [DVC](https://dvc.org/) to version datasets and model files (`*.h5`).

```bash
dvc pull   # fetch data/models tracked by DVC
dvc repro  # reproduce the pipeline
```
