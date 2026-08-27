from pathlib import Path
from unittest.mock import patch

from mattergen.common.utils.globals import DEFAULT_SAMPLING_CONFIG_PATH
from mattergen.evaluation.reference.presets import get_reference_dataset_path


def test_default_sampling_configs_are_packaged() -> None:
    assert (DEFAULT_SAMPLING_CONFIG_PATH / "default.yaml").is_file()
    assert (DEFAULT_SAMPLING_CONFIG_PATH / "csp.yaml").is_file()


def test_reference_dataset_is_downloaded_when_not_in_source_checkout(tmp_path: Path) -> None:
    downloaded_path = tmp_path / "reference.gz"
    with (
        patch("mattergen.evaluation.reference.presets._is_gzip_file", return_value=False),
        patch(
            "mattergen.evaluation.reference.presets.hf_hub_download",
            return_value=str(downloaded_path),
        ) as download,
    ):
        result = get_reference_dataset_path("reference_MP2020correction.gz")

    assert result == downloaded_path
    download.assert_called_once_with(
        repo_id="microsoft/mattergen",
        filename="data-release/alex-mp/reference_MP2020correction.gz",
    )
