"""Standalone scientific figure from the completed, frozen size-sweep scores."""
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


def render(results, output):
    source = Path(results)
    result = json.loads(source.read_text())
    kinds = sorted((k for k in result["parameters"] if k != "I0"), key=lambda k: result["parameters"][k])
    counts = np.array([result["parameters"][k] for k in kinds])
    labels = [str(n) if n < 1000 else f"{n / 1000:.1f}k" if n < 10000 else f"{n / 1000:.0f}k" for n in counts]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    blue, gray, purple, orange = "#2468a0", "#687f91", "#8250a3", "#b96b24"

    def values(model, split, metric):
        val = result["summary"][model][split][metric]
        return 100 * (1 - val) if metric == "error" else val

    def curve(ax, split, metric, color, label, style="-"):
        means = [values(kind, split, metric) for kind in kinds]
        low, high = [], []
        for kind in kinds:
            runs = [r for r in result["by_seed"] if r["model"] == kind and r["split"] == split]
            assert len(runs) == 3
            sample = [100 * (1 - r[metric]) if metric == "error" else r[metric] for r in runs]
            low.append(min(sample)); high.append(max(sample))
        ax.fill_between(counts, low, high, color=color, alpha=.15, linewidth=0)
        ax.plot(counts, means, style, color=color, marker="o", markersize=5, linewidth=1.8, label=label)

    def baseline(ax, split, metric, label):
        ax.axhline(values("I0", split, metric), color=orange, linestyle=":", linewidth=1.7, label=label)

    curve(axes[0, 0], "train", "error", gray, "Training", "--")
    curve(axes[0, 0], "validation", "error", blue, "Validation")
    baseline(axes[0, 0], "validation", "error", "Ising validation")
    axes[0, 0].set_title("Training and validation accuracy", loc="left", weight="bold", fontsize=12)
    axes[0, 0].set_ylabel("Next-opinion accuracy (%)")
    curve(axes[0, 1], "validation", "nll", blue, "Transformer")
    baseline(axes[0, 1], "validation", "nll", "Ising")
    axes[0, 1].set_title("Validation probability loss", loc="left", weight="bold", fontsize=12)
    axes[0, 1].set_ylabel("NLL (lower is better)")
    curve(axes[1, 0], "graph_seen", "error", gray, "Familiar graphs", "--")
    curve(axes[1, 0], "graph_unseen", "error", purple, "Unseen graphs")
    baseline(axes[1, 0], "graph_unseen", "error", "Ising unseen")
    axes[1, 0].set_title("Graph generalization: matched questions", loc="left", weight="bold", fontsize=12)
    axes[1, 0].set_ylabel("Next-opinion accuracy (%)")
    curve(axes[1, 1], "graph_unseen", "nll", purple, "Transformer")
    baseline(axes[1, 1], "graph_unseen", "nll", "Ising")
    axes[1, 1].set_title("Unseen-graph probability loss", loc="left", weight="bold", fontsize=12)
    axes[1, 1].set_ylabel("NLL (lower is better)")
    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_xticks(counts, labels)
        ax.set_xlabel("Trainable predictor parameters (log scale)")
        ax.grid(axis="y", alpha=.18)
        ax.legend(frameon=False, fontsize=9, loc="best")
        ax.axvline(818, color="#999999", linestyle="--", linewidth=.7, alpha=.3, zorder=0)
    sample = result["summary"][kinds[0]]
    graph_count = len({r["graph_id"] for r in result["by_unseen_graph"]})
    fig.suptitle("Model size and prediction quality", x=.075, y=.985, ha="left", fontsize=19, weight="bold")
    fig.text(.075, .943,
        f"{sample['train']['questions']} training questions  ·  {sample['validation']['questions']} validation questions  ·  "
        f"{sample['graph_unseen']['questions']} paired graph-test questions", color="#44515c", fontsize=11)
    fig.text(.075, .025,
        f"Bands: min–max across three training seeds. Scores average individual models. Dashed vertical line: 818-parameter reference.\n"
        f"The graph test uses {graph_count} unseen instances from the same graph family. Validation selects settings; assessment remains unscored.",
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
    parser.add_argument("--results", default="runs/breadth_320_size_sweep_v1/results/summary.json")
    parser.add_argument("--output", default="reports/breadth_320_size_sweep.png")
    args = parser.parse_args()
    render(args.results, args.output)
