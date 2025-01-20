# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Note that importing this module has two side effects:
1. It sets the environment variable `PROJECT_ROOT` to the root of the explorers project.
2. It registers a new resolver for OmegaConf, `eval`, which allows us to use `eval` in our config files.
"""
import os
from pathlib import Path

from omegaconf import OmegaConf

MODELS_PROJECT_ROOT = Path(__file__).resolve().parents[2]
print(f"MODELS_PROJECT_ROOT: {MODELS_PROJECT_ROOT}")

# Set environment variable PROJECT_ROOT so that hydra / OmegaConf can access it.
os.environ["PROJECT_ROOT"] = str(MODELS_PROJECT_ROOT)  # for hydra

DEFAULT_SAMPLING_CONFIG_PATH = Path(__file__).resolve().parents[3] / "sampling_conf"
PROPERTY_SOURCE_IDS = [
    "dft_mag_density",
    "dft_bulk_modulus",
    "dft_shear_modulus",
    "energy_above_hull",
    "formation_energy_per_atom",
    "space_group",
    "hhi_score",
    "ml_bulk_modulus",
    "chemical_system",
    "dft_band_gap",
]

SELECTED_ATOMIC_NUMBERS = [
    1,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    37,
    38,
    39,
    40,
    41,
    42,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    55,
    56,
    57,
    58,
    59,
    60,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
]
MAX_ATOMIC_NUM = 100
# Set `eval` resolver
def try_eval(s):
    """This is a custom resolver for OmegaConf that allows us to use `eval` in our config files
    with the syntax `${eval:'${foo} + ${bar}'}

    See:
    https://omegaconf.readthedocs.io/en/2.3_branch/how_to_guides.html#id1
    """
    try:
        return eval(s)
    except Exception as e:
        print(f"Calling eval on string {s} raised exception {e}")
        raise


OmegaConf.register_new_resolver("eval", try_eval)
