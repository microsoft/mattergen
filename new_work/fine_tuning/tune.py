import os
from pathlib import Path
from hydra import initialize, compose

from mattergen.scripts.finetune import mattergen_finetune


def tune(
    new_model_name: str,
    property_names: list[str],
    data_module: str,
):
    overrides = [
        "adapter.pretrained_name=mattergen_base",
        f"data_module={data_module}",
        *[
            f"+lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.{prop}={prop}"
            for prop in property_names
        ],
        "~trainer.logger",
        "~trainer.check_val_every_n_epoch",
        f"data_module.properties={property_names}",
        "trainer.max_epochs=200",
        "trainer.accumulate_grad_batches=8",
    ]

    config_path = "../../mattergen/conf"

    with initialize(config_path=config_path, version_base="1.1"):
        cfg = compose(config_name="finetune", overrides=overrides)

    run_dir = Path(f"new_work/finetuned_models/{new_model_name}")
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)

    mattergen_finetune(cfg)


if __name__ == "__main__":
    tune(
        new_model_name="topological",
        property_names=["topological"],
        data_module="mp_20",
    )
