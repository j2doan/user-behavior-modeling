"""Draw images/results-{light,dark}.png from outputs/benchmark/<model>/metrics.json.

    python images/plot_results.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "..", "outputs", "benchmark")
MODELS = ["baseline", "v0", "v1", "v2", "v3", "v4", "v5"]
TASKS = [("rating", "Rating MSE (lower is better)", min),
         ("read", "Read accuracy (higher is better)", max),
         ("category", "Category accuracy (higher is better)", max)]
THEMES = {
    "light": dict(bg="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e4e3df", bar="#86b6ef", best="#1c5cab"),
    "dark": dict(bg="#1a1a19", ink="#f0efec", ink2="#c3c2b7", grid="#383835", bar="#256abf", best="#9ec5f4"),
}

scores = {m: json.load(open(os.path.join(BENCH, m, "metrics.json"))) for m in MODELS}

for name, c in THEMES.items():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor=c["bg"])
    for ax, (task, title, pick) in zip(axes, TASKS):
        vals = [scores[m][task]["test"] for m in MODELS]
        best = pick(vals)
        colors = [c["best"] if v == best else c["bar"] for v in vals]
        ax.set_facecolor(c["bg"])
        bars = ax.bar(MODELS, vals, width=0.7, color=colors, edgecolor=c["bg"], linewidth=2, zorder=2)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.4f}",(b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8,
                        color=c["ink"] if v == best else c["ink2"], fontweight="bold" if v == best else "normal")
        ax.set_ylim(0, max(vals) * 1.12)
        ax.set_title(title, fontsize=10.5, color=c["ink"], loc="left")
        ax.tick_params(colors=c["ink2"], labelsize=9, length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(c["grid"])
        ax.grid(axis="y", color=c["grid"], lw=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, f"results-{name}.png"), dpi=150, facecolor=c["bg"])
    plt.close(fig)
