"""Reproducible figures saved in the run's artifact directory."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .models import MAIN_METHODS

LABELS = {"mean": "Mean", "linear": "Linear", "gpr_raw": "GPR (no embedding)",
          "fprm": "FPRM (tau=3)", "multiscale_equal": "Multiscale equal",
          "multiscale_weighted": "Multiscale weighted", "fprm_tau1": "FPRM (tau=1)",
          "fprm_tau6": "FPRM (tau=6)"}


def save(fig, dest):
    fig.savefig(dest, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_plots(summary: pd.DataFrame, records: list[dict], folder: Path, plot_seed: int = 0):
    folder.mkdir(parents=True, exist_ok=True)
    for experiment in summary.experiment.unique():
        block = summary[summary.experiment == experiment]
        for metric in ("nrmse", "rho", "total_seconds"):
            targets = list(block.target.unique())
            fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 4), squeeze=False)
            for target, ax in zip(targets, axes[0]):
                for method in MAIN_METHODS:
                    d = block[(block.target == target) & (block.method == method)].sort_values("rate")
                    if d.empty:
                        continue
                    d = d.dropna(subset=[metric + "_mean"])
                    if d.empty:
                        continue
                    ax.errorbar(d.rate * 100, d[metric + "_mean"],
                                yerr=d[metric + "_std"].fillna(0), capsize=2,
                                marker="o", markersize=3, label=LABELS[method])
                ax.set(title=f"{experiment}: {target}", xlabel="Missing (%)", ylabel=metric)
                ax.grid(alpha=0.2)
            axes[0, -1].legend(fontsize=8)
            fig.tight_layout()
            save(fig, folder / f"{experiment}_{metric}.png")
        # Delays and equal/weighted fusion shown explicitly, separate from simple baselines.
        fig, ax = plt.subplots(figsize=(7, 4))
        for method in ("fprm_tau1", "fprm", "fprm_tau6", "multiscale_equal", "multiscale_weighted"):
            d = block[block.method == method].groupby("rate").nrmse_mean.mean()
            if len(d):
                ax.plot(d.index * 100, d.values, marker="o", label=LABELS[method])
        ax.set(title=f"{experiment}: scale ablation (target-average means)",
               xlabel="Missing (%)", ylabel="NRMSE")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        save(fig, folder / f"{experiment}_ablation.png")
    for record in records:
        case = record["case"]
        if case["seed"] != plot_seed or case["rate"] != 0.5:
            continue
        data = pd.read_csv(record["predictions_file"])
        dates = pd.to_datetime(data.month)
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(dates, data.truth, color="black", alpha=0.6, linewidth=1, label="Truth (evaluation only)")
        ax.scatter(dates[data.hidden], data.truth[data.hidden], color="black", s=7, alpha=0.35, label="Hidden")
        ax.scatter(dates[~data.hidden], data.truth[~data.hidden], color="tab:green", s=9, label="Observed")
        for method in ("fprm", "multiscale_equal", "multiscale_weighted"):
            if method in data:
                ax.plot(dates, data[method], linewidth=0.9, label=LABELS[method])
        ax.set(title=f"{case['experiment']}: {case['target']}, reference={case['reference']}, seed={plot_seed}",
               xlabel="Month", ylabel="SST anomaly")
        ax.legend(fontsize=8, ncol=3)
        ax.grid(alpha=0.2)
        save(fig, folder / f"recovery_{case['experiment']}_{case['target']}_seed{plot_seed}.png")
