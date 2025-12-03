import pandas as pd
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from typing import Optional, List


# Build a single indexed reference table for fast lookup
def load_reference() -> pd.DataFrame:
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[2]

    sources = {
        "train": repo_root / "datasets/mp_20/train.csv",
        "val": repo_root / "datasets/mp_20/val.csv",
        "test": repo_root / "datasets/mp_20/test.csv",
        "total": repo_root / "new_work/fine_tuning/novel_data.csv",
    }
    dfs = []
    for origin, path in sources.items():
        df = pd.read_csv(path, index_col=0 if origin != "total" else None)
        df["origin"] = origin
        dfs.append(df)
    ref = pd.concat(dfs, ignore_index=True)
    ref = ref.drop_duplicates(subset="material_id", keep="first")
    ref = ref.set_index("material_id")
    return ref


@dataclass
class MatchInfo:
    mp_id: Optional[str]
    origin: Optional[str]
    topological: Optional[float]
    band_gap: Optional[float]

    @classmethod
    def empty(cls) -> "MatchInfo":
        return cls(mp_id=np.nan, origin=np.nan, topological=np.nan, band_gap=np.nan)


reference_df = load_reference()


def lookup_mp(mp_id: str) -> Optional[MatchInfo]:
    try:
        row = reference_df.loc[mp_id]
    except KeyError:
        return None
    topological = row.get("topological", np.nan)
    band_gap = row.get("band_gap", np.nan)
    return MatchInfo(mp_id=mp_id, origin=row["origin"], topological=topological, band_gap=band_gap)


def first_match_info(mp_matches: List[str]) -> MatchInfo:
    if not mp_matches:
        return MatchInfo.empty()
    for mp_id in mp_matches:
        info = lookup_mp(mp_id)
        if info:
            return info
    return MatchInfo.empty()


def process_matches_data(generated_path: Path) -> pd.DataFrame:
    df = pd.read_csv(generated_path)
    new_df = pd.DataFrame()
    df["mp_matches"] = df["matches_in_reference"].fillna("").str.findall(r"mp-\w+")
    df["match_info"] = df["mp_matches"].apply(first_match_info)
    new_df["first_mp_id"] = df["match_info"].apply(lambda m: m.mp_id)
    new_df["origin_mp_20"] = df["match_info"].apply(lambda m: m.origin)
    new_df["topological"] = df["match_info"].apply(lambda m: m.topological)
    new_df["band_gap"] = df["match_info"].apply(lambda m: m.band_gap)
    return new_df


def create_data_matches_for_dir(base_dir: Path):
    generated_path = base_dir / "metrics.csv"
    df = process_matches_data(generated_path)
    df.to_csv(base_dir / "labeled_data_matches.csv", index=False)
