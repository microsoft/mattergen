from pathlib import Path

import pytest

from mattergen.common.gemnet.gemnet import DEFAULT_SCALE_FILE, resolve_scale_file


def test_scale_file_defaults_to_packaged_resource() -> None:
    assert resolve_scale_file(None) == str(DEFAULT_SCALE_FILE)


def test_custom_scale_file_is_honored(tmp_path: Path) -> None:
    scale_file = tmp_path / "scales.json"

    assert resolve_scale_file(scale_file) == str(scale_file)


def test_legacy_internal_scale_file_uses_packaged_resource() -> None:
    with pytest.warns(DeprecationWarning, match="legacy internal scale_file"):
        result = resolve_scale_file(
            "/scratch/amlt_code/mattergen/common/gemnet/gemnet-dT.json"
        )

    assert result == str(DEFAULT_SCALE_FILE)
