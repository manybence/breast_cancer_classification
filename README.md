# Breast Cancer Classification (PyTorch)

End-to-end binary classification on the Breast Cancer Wisconsin dataset using a small feed-forward neural network. This project demonstrates a complete ML workflow: data acquisition, preprocessing, training, evaluation, and inference with reproducible artifacts.

## Highlights
- Clean, modular pipeline with clear separation of data, model, training, and inference.
- Reproducible training with persisted artifacts (model + scaler).
- Simple inference entry point that can be run directly from an IDE.
- Balanced dataset sampling to reduce class bias.

## Project Structure
- `data.py` data loading, preprocessing, and dataset caching
- `model.py` model definition and artifact IO
- `train.py` training and evaluation
- `inference.py` single-sample inference demo
- `data/` cached datasets
- `models/` trained artifacts

## Setup
Create a virtual environment and install dependencies:
```powershell
setup.sh
```


## Training
Run training from the project root:
```powershell
python -m train
```

This will:
- fetch the dataset (or use the cached file in `data/`)
- train the model
- print loss curves and final accuracy
- save artifacts to `models/`

## Inference
```powershell
python -m inference
```


## Notes on Reproducibility
- Data sampling and train/test split are seeded in `data.py`.
- The `StandardScaler` used during training is saved and reloaded for inference.

## Dataset
Breast Cancer Wisconsin (Diagnostic) dataset (UCI ML Repository). The dataset is downloaded via `ucimlrepo` and cached in `data/breast_cancer.csv`.

## Roadmap / Next Steps
- Add unit tests for data pipeline and model IO
- Compare performance with other models (logistic regression)

## License
MIT
