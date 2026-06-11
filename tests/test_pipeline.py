import shutil
import zipfile
from pathlib import Path

import pytest

from src.config.loader import load_experiment_config
from src.eval.runner import run_eval
from src.submission.package import package_submission
from src.train.lora import train_adapter

FIXTURE = Path(__file__).parent / "fixtures" / "mini_problems.jsonl"


@pytest.mark.slow
def test_proxy_pipeline_end_to_end(tmp_path):
    adapter_dir = tmp_path / "adapter"
    cfg = load_experiment_config(
        [
            "model=proxy",
            "train=smoke",
            f"data.path={FIXTURE}",
            f"train.output_dir={adapter_dir}",
            f"eval.output_dir={tmp_path / 'eval'}",
        ]
    )
    out = train_adapter(cfg)
    assert (Path(out) / "adapter_config.json").exists()

    result = run_eval(cfg, Path(out))
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["n"] >= 1

    zip_path = package_submission(Path(out), tmp_path / "submission.zip")
    with zipfile.ZipFile(zip_path) as z:
        assert "adapter_config.json" in z.namelist()

    shutil.rmtree(adapter_dir, ignore_errors=True)
