import numpy as np
import pandas as pd
import torch
from pathlib import Path
from mlp import MLPClassifier
from generate_embeddings import (
    extract_embeddings_from_gemnet,
    data_loader_from_structures,
    load_gemnet,
)
from mattergen.common.utils.globals import get_device


def predict_from_embeddings_file(model_path, embeddings_file, device, save_path=None):
    embeddings = np.load(embeddings_file)
    return predict_with_model(model_path, embeddings, device, save_path=save_path)


def predict_with_model(model_path, embeddings, device, save_path=None):
    X = torch.tensor(embeddings, dtype=torch.float32, device=device)

    model = MLPClassifier.load_model(model_path, map_location=device)
    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(X)
        logits = outputs.squeeze().cpu().numpy()
        predictions = (logits > 0).astype(int)

        # Save results to CSV if save_path is provided
        if save_path:
            results_df = pd.DataFrame(
                {
                    "predicted_logit": logits,
                    "predicted_label": predictions,
                }
            )
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(save_path, index=False)

        return logits, predictions


def predict_generated_structures():
    """Predicts topology labels for generated structures based on the folder structure used in this project."""

    base_path = Path("new_work/predictive_models")
    model_path = base_path / "topological_mp_20"
    model_folder = model_path / "model"

    device = get_device()

    run_names = [
        "mattergen_base",
        "non_topological_dgf2",
        "topological_dgf2",
        "topological_dgf5",
    ]

    for run_name in run_names:
        predict_from_embeddings_file(
            model_path=model_folder,
            embeddings_file=model_path / f"inference/embeddings/{run_name}/embeddings.npy",
            device=device,
            save_path=model_path / f"inference/predictions/{run_name}/predictions.csv",
        )


def predict_topology(structures, save_path=None):
    """
    Predicts topology labels for a list of pymatgen structures.
    The pretrained MatterGen model and prediction head are required.
    """

    device = get_device()
    mattergen_pretrained_path = Path("new_work/finetuned_models/topological")
    mattergen_model = load_gemnet(mattergen_pretrained_path)
    mattergen_model = mattergen_model.to(device)

    data_loader = data_loader_from_structures(structures)
    embeddings = extract_embeddings_from_gemnet(
        mattergen_model,
        device,
        data_loader,
    )

    logits, predictions = predict_with_model(
        model_path=Path("new_work/predictive_models/topological_mp_20/model"),
        embeddings=embeddings,
        device=device,
        save_path=save_path,
    )

    return logits, predictions


if __name__ == "__main__":
    predict_generated_structures()

    # To use the provided models to predict topology for a list of pymatgen structures:
    # structures = [...]  # List of pymatgen structures
    # predict_topology(structures=structures, save_path=None)
