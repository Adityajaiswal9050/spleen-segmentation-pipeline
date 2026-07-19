import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VERSION_HISTORY_FILE = "version_history.csv"
OUT_FILE = "dice_vs_version.png"


def load_history():
    with open(VERSION_HISTORY_FILE) as f:
        return list(csv.DictReader(f))


def main():
    rows = load_history()
    if not rows:
        print(f"No rows in {VERSION_HISTORY_FILE} yet -- run version_and_benchmark.py first.")
        return

    versions = [r["version"] for r in rows]
    dices = [float(r["mean_dice"]) for r in rows]

    plt.figure(figsize=(9, 5))
    plt.plot(versions, dices, marker="o", color="steelblue")
    for v, d in zip(versions, dices):
        plt.annotate(f"{d:.4f}", (v, d), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    plt.xlabel("Version")
    plt.ylabel("Validation mean Dice")
    plt.title("Dice score vs. version over time")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=150)
    print(f"Saved {OUT_FILE} ({len(rows)} tracked version(s): {versions[0]} -> {versions[-1]})")
    if len(rows) == 1:
        print("Note: version_history.csv tracking started at 1.0.2 -- the earlier 1.0.0 -> 1.0.1 "
              "transitions happened before per-run version history was recorded, so their exact "
              "timestamps aren't available. Future runs of version_and_benchmark.py will append here.")


if __name__ == "__main__":
    main()
