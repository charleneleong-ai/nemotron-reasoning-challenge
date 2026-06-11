"""Package a trained LoRA adapter directory into submission.zip (rank <= 32 enforced)."""

import json
import zipfile
from pathlib import Path

MAX_RANK = 32


def package_submission(adapter_dir: Path, out_zip: Path) -> Path:
    """Zip the adapter dir contents (flat). Validates adapter_config.json exists and r <= 32."""
    adapter_dir = Path(adapter_dir)
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_dir}")
    rank = json.loads(config_path.read_text()).get("r")
    if rank is not None and rank > MAX_RANK:
        raise ValueError(f"LoRA rank {rank} exceeds competition cap of {MAX_RANK}")

    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(adapter_dir.iterdir()):
            if f.is_file():
                z.write(f, arcname=f.name)
    return out_zip
