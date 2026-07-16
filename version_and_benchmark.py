import csv
import json
import os
import shutil

STATE_FILE = "version_state.json"
LOG_FILE = "benchmark_log.csv"
CHECKPOINT_DIR = "checkpoints"
LATEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_spleen_model.pth")

IMPROVEMENT_THRESHOLD = 0.005   # Dice must improve by at least this much to count as "better"
REGRESSION_THRESHOLD = 0.03     # Dice dropping by more than this counts as a regression


def load_state():
    if os.path.isfile(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"version": "0.0.0", "best_dice": None, "versioned_checkpoint": None}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def bump_version(version, part):
    major, minor, patch = (int(x) for x in version.split("."))
    if part == "minor":
        minor += 1
        patch = 0
    elif part == "patch":
        patch += 1
    return f"{major}.{minor}.{patch}"


def get_latest_dice():
    if not os.path.isfile(LOG_FILE):
        raise FileNotFoundError(f"{LOG_FILE} not found -- run evaluate_spleen.py first.")
    with open(LOG_FILE) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{LOG_FILE} has no data rows yet.")
    return rows[-1]


def main():
    latest = get_latest_dice()
    current_dice = float(latest["mean_dice"])
    timestamp = latest["timestamp"]

    state = load_state()
    best_dice = state["best_dice"]

    print(f"Latest run ({timestamp}): Dice = {current_dice:.4f}")

    if best_dice is None:
        # First run ever -- establish the baseline.
        new_version = "1.0.0"
        state.update({"version": new_version, "best_dice": current_dice})
        versioned_path = os.path.join(CHECKPOINT_DIR, f"spleen_model_v{new_version}.pth")
        shutil.copyfile(LATEST_CHECKPOINT, versioned_path)
        state["versioned_checkpoint"] = versioned_path
        print(f"No prior baseline -- establishing version {new_version} (Dice {current_dice:.4f})")

    elif current_dice >= best_dice + IMPROVEMENT_THRESHOLD:
        # Genuine improvement -- bump MINOR version, save a new versioned checkpoint.
        new_version = bump_version(state["version"], "minor")
        versioned_path = os.path.join(CHECKPOINT_DIR, f"spleen_model_v{new_version}.pth")
        shutil.copyfile(LATEST_CHECKPOINT, versioned_path)
        state.update({"version": new_version, "best_dice": current_dice, "versioned_checkpoint": versioned_path})
        print(f"IMPROVEMENT: Dice {best_dice:.4f} -> {current_dice:.4f}. Version bumped to {new_version}.")

    elif current_dice <= best_dice - REGRESSION_THRESHOLD:
        # Meaningful regression -- do NOT bump version, just warn loudly.
        print(f"** REGRESSION WARNING **: Dice dropped from {best_dice:.4f} to {current_dice:.4f} "
              f"(more than {REGRESSION_THRESHOLD} drop). Version stays at {state['version']}. "
              f"Investigate before trusting this checkpoint.")

    else:
        # Roughly stable -- bump PATCH only, no new versioned checkpoint needed.
        new_version = bump_version(state["version"], "patch")
        state["version"] = new_version
        print(f"Stable run (Dice {current_dice:.4f} vs best {best_dice:.4f}). "
              f"Patch version bumped to {new_version}.")

    save_state(state)
    print(f"Current tracked version: {state['version']} (best Dice: {state['best_dice']:.4f})")


if __name__ == "__main__":
    main()
