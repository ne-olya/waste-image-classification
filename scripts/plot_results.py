import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reports/results.csv")
    parser.add_argument("--output", default="assets/classification-results.png")
    args = parser.parse_args()
    data = pd.read_csv(args.input).dropna(how="all")
    if data.empty or not {"experiment", "accuracy", "macro_f1"}.issubset(data):
        raise SystemExit("Сначала выполните эксперименты и заполните accuracy/macro_f1")
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    data.set_index("experiment")[["accuracy", "macro_f1"]].plot.bar(
        ax=axes[0], color=["#477998", "#f2b134"]
    )
    axes[0].set(title="Качество классификации", ylabel="Значение", ylim=(0, 1))
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].scatter(
        data["latency_ms"],
        data["macro_f1"],
        s=data["model_size_mb"].clip(lower=1) * 8,
        color="#d1495b",
        alpha=0.75,
    )
    for _, row in data.iterrows():
        axes[1].annotate(
            row["experiment"],
            (row["latency_ms"], row["macro_f1"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1].set(title="Качество, latency и размер", xlabel="Latency, ms", ylabel="Macro F1")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)


if __name__ == "__main__":
    main()
