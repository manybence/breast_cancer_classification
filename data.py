"""
Data loading and preprocessing utilities for the Breast Cancer Classification project.


Responsibilities of this module:
- Fetch the raw dataset from a reliable public source (UCI ML Repository)
- Perform label encoding
- Balance the dataset to avoid class bias
- Split data into train / test sets with stratification
- Normalize features using statistics computed on the training set only
- Return PyTorch-ready tensors + fitted scaler for inference reuse
"""

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
from pathlib import Path

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)
DATA_PATH = DATA_DIR / "breast_cancer.csv"


def load_raw_data():
    """
    Load the raw Breast Cancer Wisconsin dataset from local cache or UCI ML Repository.

    Returns
    -------
    data : pd.DataFrame
        The raw dataset with features and target label.
    """

    if DATA_PATH.exists():
        print("Loading dataset from local cache...")
        data = pd.read_csv(DATA_PATH)
    else:
        print("Fetching dataset from UCI ML Repository...")
        dataset = fetch_ucirepo(id=17)
        X = dataset.data.features   # type: ignore
        y = dataset.data.targets['Diagnosis'].map({'B': 0, 'M': 1})     # type: ignore # Target labels: 'B' (Benign) and 'M' (Malignant)
        data = pd.concat([X, y], axis=1)
        data.to_csv(DATA_PATH, index=False)

    return data

def load_and_prepare(test_size: float = 0.2, random_state: int = 42):
    """
    Load, clean, balance, and preprocess the Breast Cancer Wisconsin dataset.


    Parameters
    ----------
    test_size : float, optional (default=0.2)
    Fraction of the dataset reserved for testing.

    random_state : int, optional (default=42)
    Seed used for reproducibility across sampling and splitting operations.

    Returns
    -------
    X_train : torch.FloatTensor
    Normalized training features.

    X_test : torch.FloatTensor
    Normalized test features.

    y_train : torch.LongTensor
    Training labels (0 = Benign, 1 = Malignant).

    y_test : torch.LongTensor
    Test labels (0 = Benign, 1 = Malignant).

    scaler : StandardScaler
    Fitted scaler instance used during training (must be reused for inference).
    """


    # ------------------------------------------------------------------
    # 1. Fetch dataset from UCI Machine Learning Repository
    # ------------------------------------------------------------------
    
    # Load from local cache if available, otherwise fetch and save for future use
    data = load_raw_data()

    # ------------------------------------------------------------------
    # 2. Handle class imbalance via undersampling
    # ------------------------------------------------------------------

    # To prevent bias toward the majority class, we construct a balanced subset.
    data_benign = data[data['Diagnosis'] == 0]
    data_malignant = data[data['Diagnosis'] == 1]

    # Sample equal number of examples from each class
    n_samples = min(len(data_benign), len(data_malignant), 200)
    data_benign = data_benign.sample(n_samples, random_state=random_state)
    data_malignant = data_malignant.sample(n_samples, random_state=random_state)
    balanced_data = pd.concat([data_benign, data_malignant])

    # Separate features and labels again
    X = balanced_data.drop('Diagnosis', axis=1)
    y = balanced_data['Diagnosis']
    feature_names = X.columns.tolist()


    # ------------------------------------------------------------------
    # 3. Train / test split with stratification
    # ------------------------------------------------------------------
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


    # ------------------------------------------------------------------
    # 4. Feature normalization
    # ------------------------------------------------------------------

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    # ------------------------------------------------------------------
    # 5. Convert to PyTorch tensors
    # ------------------------------------------------------------------

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.long)
    y_test = torch.tensor(y_test.values, dtype=torch.long)


    return X_train, X_test, y_train, y_test, scaler, feature_names