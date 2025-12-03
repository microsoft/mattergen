import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_scatter import scatter_mean
from pathlib import Path
from tqdm import tqdm
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import read
from typing import Dict

from mattergen.diffusion.lightning_module import DiffusionLightningModule
from mattergen.denoiser import GemNetTDenoiser
from mattergen.common.data.collate import collate
from mattergen.common.data.datamodule import worker_init_fn
from mattergen.common.gemnet.gemnet import ModelOutput
from mattergen.common.utils.globals import get_device
from mattergen.common.utils.eval_utils import load_model_diffusion
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.common.data.dataset import CrystalDataset
from mattergen.common.data.transform import symmetrize_lattice, set_chemical_system_string
from mattergen.common.data.dataset_transform import filter_sparse_properties


def dataset_from_cache(
    set="test",
    properties=[],
) -> CrystalDataset:
    """Loads a dataset from the mattergen dataset cache"""

    dataset = CrystalDataset.from_cache_path(
        cache_path=Path(f"datasets/cache/mp_20/{set}"),
        transforms=[symmetrize_lattice, set_chemical_system_string],
        properties=properties,
        dataset_transforms=[filter_sparse_properties],
    )
    return dataset


def structures_to_dataset(
    structures: list[Structure],
    properties={},
) -> CrystalDataset:
    """Converts a list of pymatgen Structures to a CrystalDataset"""

    pos_array = np.vstack([s.frac_coords for s in structures])
    cell_array = np.array([s.lattice.matrix for s in structures])
    atomic_numbers_array = np.concatenate([s.atomic_numbers for s in structures])
    num_atoms_array = np.array([len(s) for s in structures])
    structure_id_array = np.arange(len(structures))

    dataset = CrystalDataset(
        pos=pos_array,
        cell=cell_array,
        atomic_numbers=atomic_numbers_array,
        num_atoms=num_atoms_array,
        structure_id=structure_id_array,
        properties=properties,
        transforms=[symmetrize_lattice, set_chemical_system_string],
    )

    dataset = filter_sparse_properties(dataset)

    return dataset


def data_loader_from_dataset(dataset: CrystalDataset) -> DataLoader:
    return DataLoader(
        dataset,
        shuffle=False,
        batch_size=64,
        num_workers=4,
        worker_init_fn=worker_init_fn,
        collate_fn=collate,
    )


def data_loader_from_structures(
    structures: list[Structure],
    properties={},
) -> DataLoader:
    dataset = structures_to_dataset(
        structures,
        properties=properties,
    )

    return data_loader_from_dataset(dataset)


def load_gemnet(pretrained_path: Path = None) -> GemNetTDenoiser:
    """Loads a fine-tuned MatterGen model from a given path or the MatterGen Base model if no path is provided."""

    config_overrides = [
        "++lightning_module.diffusion_module.model.element_mask_func={_target_:'mattergen.denoiser.mask_disallowed_elements',_partial_:True}"
    ]

    if pretrained_path:
        checkpoint_info = MatterGenCheckpointInfo(
            model_path=pretrained_path.resolve(),
            config_overrides=config_overrides,
        )
    else:
        checkpoint_info = MatterGenCheckpointInfo.from_hf_hub(
            "mattergen_base",
            config_overrides=config_overrides,
        )

    lightning_module: DiffusionLightningModule = load_model_diffusion(checkpoint_info)
    model: GemNetTDenoiser = lightning_module.diffusion_module.model
    return model


def load_pmg_from_ase_extxyz(file_path: Path) -> list[Structure]:
    ase_structures = read(file_path, index=":")
    pmg_structures = [AseAtomsAdaptor.get_structure(ase_struct) for ase_struct in ase_structures]
    return pmg_structures


def extract_embeddings_from_gemnet(
    model: GemNetTDenoiser,
    device: torch.device,
    data_loader: DataLoader,
) -> Dict[str, torch.Tensor]:
    model.eval()

    embeddings_list = []

    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch = batch.to(device)
            t = torch.zeros(len(batch), device=batch.pos.device)

            output: ModelOutput = model.calc_embeddings(x=batch, t=t)

            embeddings = scatter_mean(output.node_embeddings, batch.batch, dim=0)
            embeddings_list.extend(embeddings.cpu().numpy())

    return np.array(embeddings_list)


def create_train_val_test_embeddings(
    model: GemNetTDenoiser,
    device: torch.device,
    output_folder: Path,
    properties=[],
):
    """Generates embeddings for train, val, and test sets of MatterGen to serve as data for training predictive models."""
    for set in ["train", "val", "test"]:
        dataset = dataset_from_cache(set=set, properties=properties)
        data_loader = data_loader_from_dataset(dataset=dataset)
        embeddings = extract_embeddings_from_gemnet(
            model,
            device,
            data_loader,
        )

        embeddings_path = output_folder / "embeddings" / f"{set}.npy"
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving embeddings with shape {embeddings.shape} to {embeddings_path}")
        np.save(embeddings_path, embeddings)

        if len(properties) == 1:
            prop = properties[0]
            print(f"Saving labels for property '{prop}'...")
            labels = dataset.properties[prop]
            labels_path = output_folder / "labels" / f"{set}.npy"
            labels_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving labels with shape {labels.shape} to {labels_path}")
            np.save(labels_path, labels)


def create_embeddings_for_structures(
    model: GemNetTDenoiser,
    device: torch.device,
    structures: list[Structure],
    output_file: Path,
):
    data_loader = data_loader_from_structures(structures)
    embeddings = extract_embeddings_from_gemnet(
        model,
        device,
        data_loader,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, embeddings)


def create_topological_training_data(model, device):
    """Creates training data for the topology prediction model."""

    output_folder = Path("new_work/predictive_models/topological_mp_20/training_data")

    create_train_val_test_embeddings(
        model=model,
        device=device,
        output_folder=output_folder,
        properties=["topological"],
    )


def create_embeddings_generated_structures(model, device):  #
    """Creates embeddings for generated structures based on the folder structure used in this project."""

    run_names = [
        "mattergen_base",
        "non_topological_dgf2",
        "topological_dgf2",
        "topological_dgf5",
    ]

    for run_name in run_names:
        input_file = Path(f"new_work/generated_structures/{run_name}/relaxed_crystals.extxyz")
        structures = load_pmg_from_ase_extxyz(input_file)
        output_file = Path(
            f"new_work/predictive_models/topological_mp_20/inference/embeddings/{run_name}/embeddings.npy"
        )
        create_embeddings_for_structures(
            model=model,
            device=device,
            structures=structures,
            output_file=output_file,
        )


if __name__ == "__main__":
    device = get_device()
    pretrained_path = Path("new_work/finetuned_models/topological")
    model = load_gemnet(pretrained_path)
    model = model.to(device)

    # A few examples of how to create embeddings for
    #   training data,
    #   generated structures,
    #   or any list of pymatgen structures

    # create_topological_training_data(model, device)
    # create_embeddings_generated_structures(model, device)
    # create_embeddings_for_structures(model, device, structures, output_file)
