import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from pathlib import Path
import pandas as pd

from mattergen.common.utils.globals import get_device
from mlp import MLPClassifier


default_train_params = {
    "hidden_size": 64,
    "num_layers": 2,
    "dropout_p": 0.3,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 1e-2,
    "num_epochs": 200,
    "patience": 50,
}


def create_model(input_size, **params):
    """Create a new MLPClassifier instance with given parameters"""
    model_params = {**default_train_params, **params}
    return MLPClassifier(
        input_size=input_size,
        hidden_size=model_params["hidden_size"],
        num_layers=model_params["num_layers"],
        dropout_p=model_params["dropout_p"],
    )


def load_data(inputs_folder, labels_folder):
    embeddings_train = np.load(inputs_folder / "train.npy")
    embeddings_val = np.load(inputs_folder / "val.npy")
    embeddings_test = np.load(inputs_folder / "test.npy")

    labels_train = np.load(labels_folder / "train.npy")
    labels_val = np.load(labels_folder / "val.npy")
    labels_test = np.load(labels_folder / "test.npy")

    def filter_nan(embeddings, labels):
        valid_indices = ~np.isnan(labels)
        return embeddings[valid_indices], labels[valid_indices]

    # Filter out embeddings where labels are NaN
    embeddings_train, labels_train = filter_nan(embeddings_train, labels_train)
    embeddings_val, labels_val = filter_nan(embeddings_val, labels_val)
    embeddings_test, labels_test = filter_nan(embeddings_test, labels_test)

    return (
        embeddings_train,
        labels_train,
        embeddings_val,
        labels_val,
        embeddings_test,
        labels_test,
    )


def train_model(data, device, **kwargs):
    """Train the model and return the best model (not saved to disk)"""
    params = {**default_train_params, **kwargs}
    batch_size = params["batch_size"]
    learning_rate = params["learning_rate"]
    weight_decay = params["weight_decay"]
    num_epochs = params["num_epochs"]
    patience = params["patience"]

    (
        embeddings_train,
        labels_train,
        embeddings_val,
        labels_val,
        embeddings_test,
        labels_test,
    ) = data

    # Convert to PyTorch tensors and move to device
    X_train = torch.tensor(embeddings_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(labels_train, dtype=torch.float32, device=device)
    X_val = torch.tensor(embeddings_val, dtype=torch.float32, device=device)
    y_val = torch.tensor(labels_val, dtype=torch.float32, device=device)

    # Create DataLoader for batch processing
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create model and move to device
    input_size = X_train.shape[1]
    model = create_model(input_size, **params)
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=20,
    )

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_state_dict = None

    print("Starting training...")
    for epoch in tqdm(range(num_epochs)):
        # Training phase
        model.train()
        epoch_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs.squeeze(), y_val).item()
        scheduler.step(val_loss)

        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state_dict = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Restore best weights
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    print(f"Returning trained model (not saved). Best val loss: {best_val_loss:.4f}")
    return model


def train_and_save(model_folder, training_data_folder, labels_data_folder, device, **kwargs):
    params = {**default_train_params, **kwargs}
    data = load_data(training_data_folder, labels_data_folder)
    model = train_model(data, device, **params)
    model.save_model(model_folder)
    print(f"Saved model to {model_folder}")


if __name__ == "__main__":
    base_path = Path("new_work/predictive_models")
    model_path = base_path / "topological_mp_20"

    model_folder = model_path / "model"
    training_data_folder = model_path / "training_data/embeddings"
    labels_data_folder = model_path / "training_data/labels"

    device = get_device()

    training_config = {
        "hidden_size": 64,
        "num_layers": 2,
        "dropout_p": 0.3,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "weight_decay": 1e-2,
        "num_epochs": 200,
        "patience": 50,
    }

    train_and_save(
        model_folder=model_folder,
        training_data_folder=training_data_folder,
        labels_data_folder=labels_data_folder,
        device=device,
        kwargs=training_config,
    )
