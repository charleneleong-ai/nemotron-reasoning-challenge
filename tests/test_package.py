import json
import zipfile
from pathlib import Path

import pytest

from src.submission.package import package_submission


def _make_adapter(dir_: Path, rank: int) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / "adapter_config.json").write_text(
        json.dumps({"r": rank, "peft_type": "LORA"})
    )
    (dir_ / "adapter_model.safetensors").write_bytes(b"\x00\x01")
    return dir_


class TestPackage:
    def test_zip_contains_adapter_files(self, tmp_path):
        adapter = _make_adapter(tmp_path / "ad", rank=16)
        out = package_submission(adapter, tmp_path / "submission.zip")
        with zipfile.ZipFile(out) as z:
            names = z.namelist()
        assert "adapter_config.json" in names
        assert "adapter_model.safetensors" in names

    def test_rank_above_32_rejected(self, tmp_path):
        adapter = _make_adapter(tmp_path / "ad", rank=64)
        with pytest.raises(ValueError, match="rank"):
            package_submission(adapter, tmp_path / "submission.zip")

    def test_missing_adapter_config_rejected(self, tmp_path):
        bare = tmp_path / "bare"
        bare.mkdir()
        (bare / "adapter_model.safetensors").write_bytes(b"\x00")
        with pytest.raises(FileNotFoundError):
            package_submission(bare, tmp_path / "submission.zip")
