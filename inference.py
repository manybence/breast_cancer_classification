import torch
import numpy as np
import pandas as pd

from data import load_raw_data
from model import ClassificationNet
from model import load_model



def predict(sample: np.ndarray):

    model, scaler, feature_names = load_model(ClassificationNet)
    sample_df = pd.DataFrame(
        sample.reshape(1, -1),
        columns=feature_names
    )
    sample = scaler.transform(sample_df)
    sample = torch.tensor(sample, dtype=torch.float32)          # type: ignore

    with torch.no_grad():
        logits = model(sample)
        pred = torch.argmax(logits, dim=1).item()

    probs = torch.softmax(logits, dim=1)
    confidence = probs.max().item()

    return {
        "prediction": "Malignant" if pred == 1 else "Benign", 
        "confidence": round(confidence, 4)
    }



if __name__ == "__main__":

    # ------------------------------------------------------------
    # Load dataset and extract a real test sample, then run inference
    # ------------------------------------------------------------
    data_raw = load_raw_data()
    X_raw = data_raw.drop('Diagnosis', axis=1)
    y_raw = data_raw['Diagnosis']

    # Randomly select a sample from the raw dataset for inference
    idx = np.random.randint(0, len(X_raw))

    sample_features = np.array(X_raw.iloc[idx].values)
    true_label = y_raw.iloc[idx]

    print("\nPrediction:", predict(sample_features))
    print("Ground truth:", "Malignant" if true_label == 1 else "Benign")

