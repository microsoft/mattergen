import torch
import torch.nn as nn
import json
from pathlib import Path


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_p):
        super(MLPClassifier, self).__init__()
        self.num_layers = num_layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)

        # Input layer
        layers = [nn.Linear(input_size, hidden_size), self.relu, self.dropout]

        # Hidden layers
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), self.relu, self.dropout]

        # Output layer (no activation)
        layers.append(nn.Linear(hidden_size, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def save_model(self, folder_path):
        """Save model weights and configuration"""
        folder_path = Path(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)
        config = {
            "input_size": self.model[0].in_features,
            "hidden_size": self.model[0].out_features,
            "num_layers": self.num_layers,
            "dropout_p": self.model[2].p,
        }
        with open(folder_path / "model_config.json", "w") as f:
            json.dump(config, f)
        torch.save(self.state_dict(), folder_path / "best_model.pth")

    @classmethod
    def load_model(cls, folder_path, map_location=None):
        """Load model from folder and return instance"""
        folder_path = Path(folder_path)
        with open(folder_path / "model_config.json", "r") as f:
            config = json.load(f)
        model = cls(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config.get("num_layers", 3),
            dropout_p=config.get("dropout_p", 0.3),
        )
        state = torch.load(folder_path / "best_model.pth", map_location=map_location)
        model.load_state_dict(state)
        model.eval()
        return model
