"""Render the final size-sweep figure from frozen summary values.

Kept separate so plot formatting never changes the frozen assessment code.
"""
from pathlib import Path
import sys
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/agent_interact_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, NullLocator
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write
import json


def main():
    root = Path("runs/transformer_size_sweep_v1")
    protocol = json.loads((root/"protocol.json").read_text())
    table = pd.read_csv(root/"summary.csv").set_index(["model", "split"])
    kinds = ["tiny", "medium", "small"]
    x = [protocol["parameters"][kind] for kind in kinds]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for ax, metric, title, scale in ((axes[0], "error", "Prediction error (%)", 100),
                                    (axes[1], "nll", "Negative log likelihood", 1)):
        for split, label, color in (("train", "Training", "#b5651d"),
                                    ("assessment", "Held-out questions", "#1d6394")):
            values = [scale*table.loc[(kind, split), metric] for kind in kinds]
            ax.plot(x, values, "o-", color=color, label=label, lw=2)
            for xx, yy in zip(x, values):
                ax.annotate(f"{yy:.2f}" if metric == "error" else f"{yy:.3f}",
                    (xx, yy), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": .5, "alpha": .95})
        for baseline, label, style in (("I0", "Ising", ":"),
                                       ("I0_current", "Ising + current opinion", "--")):
            ax.axhline(scale*table.loc[(baseline, "assessment"), metric], color="0.45", ls=style, label=label)
        ax.set_xscale("log")
        ax.set_xticks(x, [f"{value:,}" for value in x])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Transformer parameters")
        ax.set_ylabel(title)
        ax.grid(alpha=.15)
        ax.margins(x=.18, y=.22)
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle("Transformer size: same 20 development questions and nested folds", fontsize=12)
    fig.text(.5, .012, "Exploratory means; paired uncertainty intervals are reported in the accompanying table.",
             ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, .045, 1, 1))
    output = Path("reports/transformer_size_sweep.png")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    audit_path = root/"completion_audit.json"
    audit = json.loads(audit_path.read_text())
    audit["figure_formatting"] = "Separate renderer removes overlapping logarithmic minor-tick labels; summary values unchanged"
    audit["artifacts"][str(output)] = digest(output)
    audit["artifacts"][str(Path(__file__).relative_to(Path.cwd()))] = digest(__file__)
    audit["figure_summary_sha256"] = digest(root/"summary.csv")
    json_write(audit_path, audit)


if __name__ == "__main__":
    main()
