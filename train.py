from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from data import load_and_prepare
from model import ClassificationNet
from model import save_model



def train(epochs: int = 100, batch_size: int = 32, lr: float = 1e-3):
    """
    Train a neural network classifier on the Breast Cancer dataset.

    The function performs:
    - Data loading and preprocessing
    - Mini-batch training
    - Validation loss monitoring
    - Final test-set accuracy evaluation
    - Model + scaler persistence
    """

    # ---------------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------------
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_prepare()       # type: ignore

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
    )

    # ---------------------------------------------------------------
    # 2. Model, loss, optimizer
    # ---------------------------------------------------------------
    model = ClassificationNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)


    # ---------------------------------------------------------------
    # 3. Training loop
    # ---------------------------------------------------------------

    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):

        # Training phase
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Print training and validation loss every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:03d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()


    # ---------------------------------------------------------------
    # 5. Final evaluation: classification accuracy
    # ---------------------------------------------------------------
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(yb.cpu().numpy())

    # Calculate and print accuracy
    accuracy = correct / total
    print(f"Final Test Accuracy: {accuracy * 100:.2f}%")

    # Plot confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "Malignant"]
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix – Test Set")
    plt.show()


    # ---------------------------------------------------------------
    # 6. Persist model and preprocessing artifacts
    # ---------------------------------------------------------------
    save_model(model, scaler, feature_names) 




if __name__ == "__main__":
    train()
