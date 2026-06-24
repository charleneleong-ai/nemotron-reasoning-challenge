"""One-shot: materialize ~/.kaggle/kaggle.json from KAGGLE_API_TOKEN in .env, then submit.

The Kaggle CLI 2.2.1 reads ~/.kaggle/kaggle.json (or KAGGLE_USERNAME+KAGGLE_KEY), not
KAGGLE_API_TOKEN. .env carries the credential as KAGGLE_API_TOKEN — either the full
kaggle.json JSON blob, or a bare key (then KAGGLE_USERNAME must also be set). This never
prints the secret. Usage: python scripts/_kaggle_submit.py <zip> <message>
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.submission.submit import submit

load_dotenv()
tok = os.environ.get("KAGGLE_API_TOKEN", "").strip()
if not tok:
    sys.exit("KAGGLE_API_TOKEN not set in .env")

cfg = Path.home() / ".kaggle" / "kaggle.json"
cfg.parent.mkdir(parents=True, exist_ok=True)

if tok.startswith("{"):
    creds = json.loads(tok)
elif os.environ.get("KAGGLE_USERNAME"):
    creds = {"username": os.environ["KAGGLE_USERNAME"], "key": tok}
else:
    sys.exit(
        "KAGGLE_API_TOKEN is not JSON and KAGGLE_USERNAME is unset — cannot build kaggle.json"
    )

if not creds.get("username") or not creds.get("key"):
    sys.exit("kaggle credential missing username/key")

cfg.write_text(json.dumps(creds))
cfg.chmod(stat.S_IRUSR | stat.S_IWUSR)
print(f"wrote {cfg} for user {creds['username']!r}", flush=True)

submit(Path(sys.argv[1]), sys.argv[2])
