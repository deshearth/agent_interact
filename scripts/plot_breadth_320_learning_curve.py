"""Standalone figure from completed frozen learning-curve scores."""
from argparse import ArgumentParser
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(source, output):
    source = Path(source)
    result = json.loads(source.read_text())
    counts = sorted(map(int, result["summary"]))
    kinds = sorted((k for k in result["parameters"] if k != "I0"), key=lambda k: result["parameters"][k])
    colors = {"nano": "#2468a0", "tiny": "#259d77", "wide": "#8e5aa9", "I0": "#b96b24"}
    labels = {k: f"{result['parameters'][k]:,} parameters" for k in kinds}
    labels["I0"] = "Ising (19 coefficients)"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))

    def curve(ax, kind, split, metric):
        values, lo, hi = [], [], []
        for n in counts:
            row = result["summary"][str(n)][kind][split]
            lower, upper = row["subset_ranges"][metric]
            if metric == "error":
                values.append(100 * (1 - row[metric])); lo.append(100 * (1 - upper)); hi.append(100 * (1 - lower))
            else:
                values.append(row[metric]); lo.append(lower); hi.append(upper)
        ax.fill_between(counts, lo, hi, color=colors[kind], alpha=.12, linewidth=0)
        ax.plot(counts, values, color=colors[kind], linestyle=":" if kind == "I0" else "-",
                marker="o", markersize=4.5, linewidth=1.8, label=labels[kind])

    for kind in kinds + ["I0"]:
        curve(axes[0, 0], kind, "validation", "error")
        curve(axes[0, 1], kind, "validation", "nll")
        curve(axes[1, 0], kind, "train", "error")
    axes[0, 0].set_title("Validation accuracy", loc="left", weight="bold", fontsize=12)
    axes[0, 0].set_ylabel("Next-opinion accuracy (%)")
    axes[0, 1].set_title("Validation probability loss", loc="left", weight="bold", fontsize=12)
    axes[0, 1].set_ylabel("NLL (lower is better)")
    axes[1, 0].set_title("Training accuracy", loc="left", weight="bold", fontsize=12)
    axes[1, 0].set_ylabel("Next-opinion accuracy (%)")
    reference = kinds[0]
    for kind in kinds[1:]:
        means, lo, hi = [], [], []
        for n in counts:
            a = {r["fold"]: r["error"] for r in result["by_subset"]
                 if r["n_questions"] == n and r["model"] == kind and r["split"] == "validation"}
            b = {r["fold"]: r["error"] for r in result["by_subset"]
                 if r["n_questions"] == n and r["model"] == reference and r["split"] == "validation"}
            assert set(a) == set(b)
            delta = np.array([100 * (b[f] - a[f]) for f in sorted(a)])
            means.append(delta.mean()); lo.append(delta.min()); hi.append(delta.max())
        axes[1, 1].fill_between(counts, lo, hi, color=colors[kind], alpha=.12, linewidth=0)
        axes[1, 1].plot(counts, means, color=colors[kind], marker="o", markersize=4.5, linewidth=1.8, label=labels[kind])
    axes[1, 1].axhline(0, color="#7c8791", linestyle="--", linewidth=1)
    axes[1, 1].set_title(f"Accuracy gain over {result['parameters'][reference]:,} parameters", loc="left", weight="bold", fontsize=12)
    axes[1, 1].set_ylabel("Validation difference (percentage points)")
    for ax in axes.flat:
        ax.set_xscale("log", base=2)
        ax.set_xticks(counts, [str(n) for n in counts])
        ax.set_xlabel("Training questions (log scale)")
        ax.grid(axis="y", alpha=.18)
        ax.legend(frameon=False, fontsize=9, loc="best")
    validation_count = result["summary"][str(counts[-1])][reference]["validation"]["questions"]
    path_count = result["summary"][str(counts[0])][reference]["validation"]["subset_paths"]
    seeds = {r["seed"] for r in result["by_seed"] if r["model"] == reference}
    fig.suptitle("Training questions and model size", x=.075, y=.982, ha="left", fontsize=19, weight="bold")
    fig.text(.075, .941,
        f"Same {validation_count} validation questions  ·  {path_count} nested subset paths  ·  {len(seeds)} training seeds per condition",
        fontsize=11, color="#44515c")
    fig.text(.075, .025,
        f"Bands span subset-path means after averaging training seeds; they are not confidence intervals. The {counts[-1]}-question endpoint is shared once.\n"
        "Question PCA and settings are fitted within each training condition. Validation is reused; assessment remains unscored.",
        fontsize=9, color="#44515c", linespacing=1.5)
    fig.tight_layout(rect=(.045, .085, .995, .92), h_pad=2., w_pad=2.3)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    record = {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "plot_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "figure": str(output), "figure_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--results", default="runs/breadth_320_learning_curve_v1/results/summary.json")
    parser.add_argument("--output", default="reports/breadth_320_learning_curve.png")
    args = parser.parse_args()
    render(args.results, args.output)
