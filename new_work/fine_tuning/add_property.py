import pandas as pd

TOPOLOGY_CATEGORIES = {
    "LCEBR": 0,  # trivial
    "ES": 1,
    "ESFD": 2,
    "SEBR": 3,
    "NLC": 4,
}


# def transform_data():
#     topology_file_path = "new_work/fine_tuning/novel_data.csv"
#     topology_df = pd.read_csv(topology_file_path, index_col=0)
#     topology_df["topology_category"] = topology_df["topology_label"].map(TOPOLOGY_CATEGORIES)
#     topology_df["topological"] = (
#         topology_df["topology_category"].map(lambda x: x != 0, na_action="ignore").astype(float)
#     )
#     topology_df.to_csv(topology_file_path, index=True)


def add_topology_category_to_csv(csv_file_path, topology_file_path, dry_run=False):
    df = pd.read_csv(csv_file_path, index_col=0)
    orginial_df = df.copy()
    topology_df = pd.read_csv(topology_file_path)
    topology_df = topology_df[["material_id", "band_gap", "topology_category", "topological"]]

    # drop topology columns from df if they exist
    df.drop(columns=["band_gap", "topology_category", "topological"], errors="ignore", inplace=True)

    # Merge the dataframes on the "material_id" column
    df = df.merge(topology_df, on="material_id", how="left")

    # Print number of rows where topology_category was successfully added
    print(f"Number of rows with topology_category: {df['topology_category'].notnull().sum()}")

    if dry_run:
        print("Dry run: no changes made to the CSV file.")
        print(f"Would cause changes: {not df.equals(orginial_df)}")
        return

    df.to_csv(path, index=True)
    print("Topology category added to CSV file.")


if __name__ == "__main__":
    datasets = ["train", "test", "val"]
    for dataset in datasets:
        path = f"datasets/mp_20/{dataset}.csv"
        topology_file_path = "new_work/fine_tuning/novel_data.csv"
        add_topology_category_to_csv(path, topology_file_path, dry_run=False)
