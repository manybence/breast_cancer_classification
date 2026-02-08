import torch
import joblib
import torch.nn as nn

model_path = "./models/artifacts_model.pt"
scaler_path = "./models/artifacts_scaler.pkl"


class ClassificationNet(nn.Module):
    """
    Simple feed-forward neural network for binary classification.

    Architecture:
    - Input layer: 30 features (Breast Cancer Wisconsin dataset)
    - Hidden layer: 64 units + ReLU
    - Output layer: 2 logits (Benign / Malignant)
    """
        
    def __init__(self, input_units=30, hidden_units=64, output_units=2):
        super().__init__()
        self.fc1 = nn.Linear(input_units, hidden_units)
        self.fc2 = nn.Linear(hidden_units, output_units)


    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        return self.fc2(x)
    

def save_model(model, scaler, feature_names, path_prefix="./models/artifacts"):
    """
    Persist the trained model and preprocessing objects to disk.

    Saving them together ensures inference uses the *exact same*
    transformations as training.
    """
    torch.save(model.state_dict(), f"{path_prefix}_model.pt")
    joblib.dump(scaler, f"{path_prefix}_scaler.pkl")
    joblib.dump(feature_names, f"{path_prefix}_features.pkl")


def load_model(model_class, path_prefix="./models/artifacts"):
    """
    Load a trained model and its associated scaler for inference.
    """
    model = model_class()
    model.load_state_dict(torch.load(f"{path_prefix}_model.pt"))
    model.eval()  # important: disables dropout, batchnorm updates, etc.

    scaler = joblib.load(f"{path_prefix}_scaler.pkl")
    feature_names = joblib.load(f"{path_prefix}_features.pkl")

    return model, scaler, feature_names
