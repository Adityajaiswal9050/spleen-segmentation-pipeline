import json
import os
import csv
import shutil
import pytest

import version_and_benchmark as vb


@pytest.fixture
def isolated_workdir(tmp_path, monkeypatch):
    """Run each test inside a throwaway temp directory so tests never
    touch the real benchmark_log.csv / version_state.json / checkpoints."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("checkpoints", exist_ok=True)
    # a tiny fake "checkpoint" file -- content doesn't matter, only that it exists
    with open(os.path.join("checkpoints", "best_spleen_model.pth"), "wb") as f:
        f.write(b"fake-weights")
    return tmp_path


def write_log(rows):
    with open(vb.LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "checkpoint", "num_val_cases", "mean_dice"])
        for row in rows:
            writer.writerow(row)


def test_bump_version_minor():
    assert vb.bump_version("1.2.3", "minor") == "1.3.0"


def test_bump_version_patch():
    assert vb.bump_version("1.2.3", "patch") == "1.2.4"


def test_first_run_establishes_baseline(isolated_workdir):
    write_log([["2026-01-01T00:00:00", "checkpoints/best_spleen_model.pth", 8, "0.4000"]])
    vb.main()
    state = json.load(open(vb.STATE_FILE))
    assert state["version"] == "1.0.0"
    assert state["best_dice"] == 0.4000
    assert os.path.isfile("checkpoints/spleen_model_v1.0.0.pth")


def test_improvement_bumps_minor_version(isolated_workdir):
    # Seed an existing state as if version 1.0.0 already exists with Dice 0.40
    json.dump({"version": "1.0.0", "best_dice": 0.40, "versioned_checkpoint": "checkpoints/spleen_model_v1.0.0.pth"},
               open(vb.STATE_FILE, "w"))
    write_log([["2026-01-02T00:00:00", "checkpoints/best_spleen_model.pth", 8, "0.4700"]])  # +0.07 improvement
    vb.main()
    state = json.load(open(vb.STATE_FILE))
    assert state["version"] == "1.1.0"
    assert state["best_dice"] == 0.47
    assert os.path.isfile("checkpoints/spleen_model_v1.1.0.pth")


def test_regression_does_not_bump_version(isolated_workdir):
    json.dump({"version": "1.1.0", "best_dice": 0.47, "versioned_checkpoint": "checkpoints/spleen_model_v1.1.0.pth"},
               open(vb.STATE_FILE, "w"))
    write_log([["2026-01-03T00:00:00", "checkpoints/best_spleen_model.pth", 8, "0.3900"]])  # big drop (>0.03)
    vb.main()
    state = json.load(open(vb.STATE_FILE))
    # Version must NOT change on a regression
    assert state["version"] == "1.1.0"
    assert state["best_dice"] == 0.47


def test_stable_run_bumps_patch_only(isolated_workdir):
    json.dump({"version": "1.1.0", "best_dice": 0.47, "versioned_checkpoint": "checkpoints/spleen_model_v1.1.0.pth"},
               open(vb.STATE_FILE, "w"))
    write_log([["2026-01-04T00:00:00", "checkpoints/best_spleen_model.pth", 8, "0.4705"]])  # negligible change
    vb.main()
    state = json.load(open(vb.STATE_FILE))
    assert state["version"] == "1.1.1"
    # best_dice should stay at the old best since this wasn't a real improvement
    assert state["best_dice"] == 0.47
