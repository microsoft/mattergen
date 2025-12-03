from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from pathlib import Path
import pandas as pd


def in_mp(list_str: str) -> bool:
    if isinstance(list_str, float):
        return False

    vals = list_str.strip("[]").replace("'", "").replace('"', "").split(", ")

    for v in vals:
        if v.startswith("mp"):
            return True
    return False


def aggregate(generated_structures_dir: Path, predictions_path: Path | None) -> pd.DataFrame:
    structures_path = generated_structures_dir / "relaxed_crystals.extxyz"
    metrics_path = generated_structures_dir / "metrics.csv"
    matched_labels_path = generated_structures_dir / "labeled_data_matches.csv"
    space_group_info_path = generated_structures_dir / "space_group_info.csv"

    structures = read(structures_path, index=":")
    structures = [AseAtomsAdaptor.get_structure(s) for s in structures]

    if not (matched_labels_path).exists():
        from labeled_data_matches import create_data_matches_for_dir

        create_data_matches_for_dir(generated_structures_dir)

    if not (space_group_info_path).exists():
        from space_groups import create_space_group_info

        create_space_group_info(structures, generated_structures_dir)

    for path in [metrics_path, matched_labels_path, space_group_info_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file {path} not found.")

    metrics = pd.read_csv(metrics_path)
    matched_labels = pd.read_csv(matched_labels_path)
    space_group_info = pd.read_csv(space_group_info_path)

    all_data = pd.concat(
        [
            metrics,
            matched_labels,
            space_group_info,
        ],
        axis=1,
    )

    # add predictions if available
    if predictions_path and predictions_path.exists():
        predictions = pd.read_csv(predictions_path)
        all_data = pd.concat([all_data, predictions], axis=1)

    all_data["structure"] = structures

    # calculate number of distinct elements in the composition
    all_data["num_elements"] = all_data["structure"].apply(lambda s: len(s.composition.elements))

    # contains at least one p-block element
    all_data["p-block"] = all_data["structure"].apply(
        lambda s: any(el.block == "p" for el in s.composition.elements)
    )

    all_data["in_mp"] = all_data["matches_in_reference"].apply(in_mp)

    return all_data


if __name__ == "__main__":
    print(aggregate("mattergen_base").head())
