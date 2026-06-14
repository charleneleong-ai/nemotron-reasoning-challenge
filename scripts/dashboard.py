"""Self-contained progress dashboard for the Nemotron solver-SFT effort.

Snapshots current state into a single HTML page (no external deps, inline SVG):
  - live SFT training loss curve + step progress (parsed from the run log)
  - per-category corpus coverage (verified solver traces vs gaps)
  - Kaggle score history
Serves it at http://localhost:<port>/progress.html.

    .venv/bin/python scripts/dashboard.py            # build + serve
    .venv/bin/python scripts/dashboard.py --no-serve # just write progress.html
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import typer

from src.data.categories import classify
from src.solve.registry import SOLVERS

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

_LOSS = re.compile(r"'loss': ([0-9.]+)")
_STEP = re.compile(r"(\d+)/(\d+) \[")

# Kaggle public scores so far (description, score, note).
SCORES = [
    ("CoT-SFT 2-epoch (generic traces)", 0.62, "best so far"),
    ("CoT-SFT 1-epoch (generic traces)", 0.61, ""),
    ("Smoke (10-step)", 0.58, ""),
    ("GRPO from base (Prime hosted)", 0.49, "RL from base hurt"),
]


def parse_log(path: Path) -> tuple[list[float], int, int]:
    losses: list[float] = []
    cur = total = 0
    if path.exists():
        text = path.read_text(errors="ignore")
        losses = [float(x) for x in _LOSS.findall(text)]
        steps = _STEP.findall(text)
        if steps:
            cur, total = int(steps[-1][0]), int(steps[-1][1])
    return losses, cur, total


def corpus_stats(path: Path) -> list[tuple[str, int, int]]:
    """Return (category, verified, total) sorted by total desc."""
    verified: dict[str, int] = {}
    total: dict[str, int] = {}
    if path.exists():
        for line in path.open():
            r = json.loads(line)
            cat = classify(r["prompt"]) or "unknown"
            total[cat] = total.get(cat, 0) + 1
            s = SOLVERS.get(cat)
            if s and (s.solve(r["prompt"]) or "").strip() == str(r["answer"]).strip():
                verified[cat] = verified.get(cat, 0) + 1
    return sorted(
        ((c, verified.get(c, 0), n) for c, n in total.items()),
        key=lambda x: -x[2],
    )


def _sparkline(losses: list[float], w: int = 720, h: int = 220) -> str:
    if len(losses) < 2:
        return "<p>waiting for training steps…</p>"
    lo, hi = min(losses), max(losses)
    span = hi - lo or 1.0
    pts = " ".join(
        f"{w * i / (len(losses) - 1):.1f},{h - (h - 20) * (v - lo) / span - 10:.1f}"
        for i, v in enumerate(losses)
    )
    return (
        f'<svg width="{w}" height="{h}" style="background:#0d1117;border-radius:8px">'
        f'<polyline points="{pts}" fill="none" stroke="#58a6ff" stroke-width="2"/>'
        f'<text x="8" y="16" fill="#8b949e" font-size="12">loss {hi:.2f}</text>'
        f'<text x="8" y="{h - 6}" fill="#8b949e" font-size="12">loss {lo:.2f}</text>'
        f"</svg>"
    )


def _table(rows: list[tuple], headers: tuple[str, ...]) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render(log: Path, corpus: Path) -> str:
    losses, cur, total = parse_log(log)
    stats = corpus_stats(corpus)
    verified_total = sum(v for _, v, _ in stats)
    grand_total = sum(t for _, _, t in stats)
    pct = (100 * cur / total) if total else 0
    last = f"{losses[-1]:.3f}" if losses else "—"

    cov_rows = [
        (c, v, t, f"{100 * v / t:.0f}%" if t else "—", t - v) for c, v, t in stats
    ]
    score_rows = [(d, f"{s:.2f}", n) for d, s, n in SCORES]

    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="30">
<title>Nemotron solver-SFT progress</title>
<style>
body{{font-family:ui-monospace,Menlo,monospace;background:#0d1117;color:#c9d1d9;margin:24px;max-width:840px}}
h1{{font-size:18px}} h2{{font-size:14px;color:#8b949e;margin-top:28px;border-bottom:1px solid #30363d;padding-bottom:4px}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{text-align:left;padding:4px 10px;border-bottom:1px solid #21262d}}
th{{color:#8b949e}} .big{{font-size:15px;color:#58a6ff}} .bar{{height:8px;background:#21262d;border-radius:4px;overflow:hidden}}
.fill{{height:8px;background:#3fb950}}
</style></head><body>
<h1>🦥 Nemotron solver-SFT progress <span style="color:#8b949e;font-size:12px">(auto-refresh 30s)</span></h1>

<h2>Local SFT run (hedge)</h2>
<p class="big">step {cur}/{total} &nbsp; ({pct:.1f}%) &nbsp; latest loss {last}</p>
<div class="bar"><div class="fill" style="width:{pct:.1f}%"></div></div>
{_sparkline(losses)}

<h2>Corpus coverage — {verified_total}/{grand_total} verified-correct traces</h2>
{_table(cov_rows, ("category", "verified", "total", "%", "gap"))}

<h2>Kaggle scores</h2>
{_table(score_rows, ("submission", "score", "note"))}
</body></html>"""


@app.command()
def main(
    log: str = typer.Option(
        "", help="SFT run log (blank => latest logs/sft_run_*.log)."
    ),
    corpus: str = typer.Option("data/cot_hybrid.jsonl"),
    out: str = typer.Option("progress.html"),
    port: int = typer.Option(8011),
    serve: bool = typer.Option(True),
) -> None:
    def latest_log() -> Path:
        return (
            Path(log)
            if log
            else Path(max(glob.glob("logs/sft_run_*.log"), default="logs/_none.log"))
        )

    Path(out).write_text(render(latest_log(), Path(corpus)))
    print(f"wrote {out} (log={latest_log().name})", flush=True)
    if not serve:
        return

    import http.server
    import socketserver

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 — stdlib handler override; regenerate live
            html = render(latest_log(), Path(corpus)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)

        def log_message(self, *_: object) -> None:
            pass

    print(f"serving http://localhost:{port}/ (live)", flush=True)
    with socketserver.TCPServer(("", port), Handler) as s:
        s.serve_forever()


if __name__ == "__main__":
    app()
