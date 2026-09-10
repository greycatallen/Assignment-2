"""Execute the classroom notebook in order and check actual training artifacts.

Run with the same Python as your notebook kernel:
    python tests/verify_notebook.py --kernel py313
Only the in-memory test copy uses five episodes and samples every two games.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys
import zipfile

import nbformat
from nbclient import NotebookClient
from PIL import Image, ImageSequence
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pacman_player import PREVIEW_PLAYS, _advance


def verify():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="py313")
    parser.add_argument("--no-popups", action="store_true")
    args = parser.parse_args()
    playback = {"index": 0, "paused": False, "completed": 0}
    for _ in range(3 * PREVIEW_PLAYS):
        _advance(playback, 3)
    assert playback == {"index": 2, "paused": True, "completed": PREVIEW_PLAYS}
    root = ROOT
    notebook = nbformat.read(root / "pacman_dqn.ipynb", as_version=4)
    for cell in notebook.cells:
        tags = cell.metadata.get("tags", [])
        if "choices" in tags:
            cell.source = re.sub(r"^EPISODES = \d+$", "EPISODES = 5",
                                 cell.source, flags=re.MULTILINE)
        if "settings" in tags:
            cell.source = cell.source.replace("DEMO_EVERY = 25", "DEMO_EVERY = 2")
        if args.no_popups and "preview-settings" in tags:
            cell.source = cell.source.replace("SHOW_POPUPS = True", "SHOW_POPUPS = False")
    notebook.cells.append(nbformat.v4.new_code_cell('print("VERIFIED_RUN=" + str(RUN_DIR.resolve()))'))
    client = NotebookClient(notebook, timeout=600, kernel_name=args.kernel,
                            resources={"metadata": {"path": str(root)}})
    client.execute()
    output = "".join(o.get("text", "") for o in notebook.cells[-1].outputs)
    run = Path(output.split("VERIFIED_RUN=", 1)[1].strip())
    nbformat.write(notebook, run / "executed_verification.ipynb")

    summary = json.loads((run / "training_summary.json").read_text())
    assert summary["status"] == "completed" and summary["completed_episodes"] == 5
    assert summary["learning_updates"] > 0
    config = json.loads((run / "config.json").read_text())
    assert config["python"].startswith("3.13.") and config["preview_speed"] == 4
    assert config["preview_plays"] == PREVIEW_PLAYS == 2
    with (run / "training.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    assert all(math.isfinite(float(row["mean_loss"])) for row in rows if int(row["total_steps"]) >= 1000)

    before = torch.load(run / "untrained.pt", weights_only=True)
    after = torch.load(run / "trained.pt", weights_only=True)
    assert any(not torch.equal(before["model"][k], v) for k, v in after["model"].items())
    assert all(torch.isfinite(v).all() for v in after["model"].values())
    comparison = json.loads((run / "comparison.json").read_text())
    assert comparison["before"]["seeds"] == comparison["after"]["seeds"]
    assert len(comparison["before"]["scores"]) == len(comparison["after"]["scores"]) == 5
    demos = json.loads((run / "demo_scores.json").read_text())
    assert [demo["episode"] for demo in demos] == [2, 4]
    for episode in (2, 4):
        assert (run / f"episode_{episode:04d}.pt").is_file()
    for name in ("episode_0000.gif", "episode_0002.gif", "episode_0004.gif", "final_best.gif"):
        with Image.open(run / "demos" / name) as gif:
            count = gif.n_frames
            assert gif.info.get("loop") == PREVIEW_PLAYS - 1
            duration = sum(f.info.get("duration", 0) for f in ImageSequence.Iterator(gif))
            assert 1 < count <= 75, (name, count)
            assert 0 < duration <= 5000, (name, duration)
    with Image.open(run / "training_dashboard.png") as plot:
        plot.verify()
    with zipfile.ZipFile(str(run) + ".zip") as archive:
        assert archive.testzip() is None
        assert "comparison.json" in archive.namelist()
    print(json.dumps({"run": str(run), "config": config, "summary": summary}, indent=2))
    print("PASS: ordered notebook execution, learning, periodic samples, GIF speed, checkpoints, comparison, plot, ZIP")


if __name__ == "__main__":
    verify()
