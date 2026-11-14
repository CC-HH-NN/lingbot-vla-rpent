"""Generate the portfolio figures from the sanitized result tables.

The script reads only files under results/ and writes SVG/PNG figures under
assets/figures/. It never reads model, server, or runtime paths.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "assets" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# 使用 Okabe-Ito 风格配色，并在图例和位置上同时保留文字标识。
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
INK = "#1F2937"
MUTED = "#5B6675"
GRID = "#D8DEE8"
PHASE_COLORS = {"SFT": BLUE, "Enhanced": ORANGE, "ActionEMA": PURPLE, "B": GREEN}
CHECKPOINT_NAMES = {
    "M0": "SFT 16k",
    "M1": "Enhanced 23k",
    "M2": "ActionEMA 26k",
    "M4": "B 29k",
    "C_student": "C student 29k",
    "C_teacher": "C teacher 29k",
    "Agent": "Official Agent",
}


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#8B95A5",
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    svg_path = FIGURES / f"{stem}.svg"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    # Matplotlib 生成的 SVG 属性可能带行尾空格，统一清理以通过仓库检查。
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(FIGURES / f"{stem}.png", format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def checkpoint_figure() -> None:
    rows = read_csv("checkpoint_summary.csv")
    checkpoints = ["M0", "M1", "M2", "M4", "C_student", "C_teacher"]
    settings = ["demo_clean", "demo_randomized"]
    values = {(r["checkpoint"], r["setting"]): float(r["success_rate"]) * 100 for r in rows}
    fig, ax = plt.subplots(figsize=(9.4, 4.9), layout="constrained")
    x = list(range(len(checkpoints)))
    width = 0.34
    for offset, setting, color, label in [
        (-width / 2, "demo_clean", BLUE, "clean"),
        (width / 2, "demo_randomized", ORANGE, "randomized"),
    ]:
        bars = ax.bar([i + offset for i in x], [values[(c, setting)] for c in checkpoints], width, color=color, label=label)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2, f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_title("Checkpoint selection on the same 250-episode protocol")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 100)
    ax.set_xticks(x, [CHECKPOINT_NAMES[c] for c in checkpoints])
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, loc="upper left")
    ax.text(0, -0.19, "Each checkpoint: 125 clean + 125 randomized episodes. Values are descriptive episode rates.", transform=ax.transAxes, color=MUTED, fontsize=9)
    save(fig, "checkpoint-success")


def inference_figure() -> None:
    rows = read_csv("inference_comparison.csv")
    names = [r["strategy"] for r in rows]
    rates = [float(r["success_rate"]) * 100 for r in rows]
    requests = [int(r["requests"]) for r in rows]
    colors = [GREEN if n == "Fixed 20" else (PURPLE if n == "AutoHorizon" else BLUE) for n in names]
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 7.0), sharex=True, layout="constrained", height_ratios=[1, 1])
    x = list(range(len(names)))
    bars = axes[0].bar(x, rates, color=colors, width=0.64)
    for bar, rate in zip(bars, rates):
        axes[0].text(bar.get_x() + bar.get_width() / 2, rate + 1.1, f"{rate:.1f}%", ha="center", va="bottom", fontsize=9)
    axes[0].axhline(rates[0], color=BLUE, linestyle="--", linewidth=1, alpha=0.7)
    axes[0].text(len(names) - 0.1, rates[0] + 1.5, "Fixed 50 baseline", ha="right", color=BLUE, fontsize=8)
    axes[0].set_ylabel("Success rate (%)")
    axes[0].set_ylim(0, 78)
    axes[0].set_title("Inference strategy comparison: feedback versus request cost")
    axes[0].grid(axis="y", color=GRID, linewidth=0.8)
    axes[0].set_axisbelow(True)
    bars = axes[1].bar(x, requests, color=["#9BBBD2" if n != "Fixed 20" else GREEN for n in names], width=0.64)
    for bar, count in zip(bars, requests):
        axes[1].text(bar.get_x() + bar.get_width() / 2, count + max(requests) * 0.025, f"{count:,}", ha="center", va="bottom", fontsize=9)
    axes[1].set_ylabel("Model requests")
    axes[1].set_xticks(x, names, rotation=12, ha="right")
    axes[1].grid(axis="y", color=GRID, linewidth=0.8)
    axes[1].set_axisbelow(True)
    axes[1].text(0, -0.36, "Same B29000 checkpoint and 250-episode protocol. Request count is an observed system cost, not a FLOP estimate.", transform=axes[1].transAxes, color=MUTED, fontsize=9)
    save(fig, "inference-strategies")


def rolling_mean(values: list[float], window: int = 101) -> list[float]:
    out = []
    total = 0.0
    queue: list[float] = []
    for value in values:
        queue.append(value)
        total += value
        if len(queue) > window:
            total -= queue.pop(0)
        out.append(total / len(queue))
    return out


def training_figure() -> None:
    rows = read_csv("training_full.csv")
    groups: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        groups[row["phase"]].append({"step": int(row["step"]), "loss": float(row["loss"]), "vla": float(row["vla_loss"])})
    order = ["SFT", "Enhanced", "ActionEMA", "B"]
    labels = {"SFT": "SFT 0–16k", "Enhanced": "Augmented 16–23k", "ActionEMA": "ActionEMA 23–26k", "B": "B route 26–29k"}
    fig, axes = plt.subplots(2, 1, figsize=(10.4, 7.4), sharex=True, layout="constrained")
    for ax, key, ylabel in [(axes[0], "loss", "total loss"), (axes[1], "vla", "VLA loss")]:
        for phase in order:
            group = sorted(groups[phase], key=lambda item: item["step"])
            xs = [item["step"] for item in group]
            ys = [item[key] for item in group]
            # 保留归档日志缺口，不连接相邻记录之间的非连续区段。
            segments = []
            start = 0
            for index in range(1, len(xs)):
                if xs[index] - xs[index - 1] > 500:
                    segments.append((xs[start:index], ys[start:index]))
                    start = index
            segments.append((xs[start:], ys[start:]))
            for segment_xs, segment_ys in segments:
                ax.plot(segment_xs, segment_ys, color=PHASE_COLORS[phase], linewidth=0.65, alpha=0.18)
                smooth = rolling_mean(segment_ys)
                ax.plot(segment_xs, smooth, color=PHASE_COLORS[phase], linewidth=2.0, label=labels[phase] if segment_xs is segments[0][0] else None)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", color=GRID, linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0.004)
    axes[0].set_title("Full archived training trace: SFT to B29000")
    axes[0].legend(ncol=2, loc="upper right", fontsize=8)
    axes[1].set_xlabel("Optimizer step")
    axes[1].set_xlim(0, 29000)
    for boundary in (16000, 23000, 26000):
        for ax in axes:
            ax.axvline(boundary, color="#7B8794", linewidth=0.8, linestyle="--")
    save(fig, "training-loss-full")


def heatmap_figure() -> None:
    rows = read_csv("per_task_checkpoint.csv")
    tasks = sorted({r["task"] for r in rows})
    checkpoints = ["M0", "M1", "M2", "M4", "Agent"]
    values = {(r["task"], r["checkpoint"]): float(r["success_rate"]) for r in rows}
    matrix = [[values[(task, ck)] for ck in checkpoints] for task in tasks]
    fig_height = max(7.0, len(tasks) * 0.34)
    fig, ax = plt.subplots(figsize=(9.8, fig_height), layout="constrained")
    image = ax.imshow(matrix, cmap="Blues", norm=Normalize(0, 1), aspect="auto")
    ax.set_title("Per-task success rate across checkpoints and the official Agent")
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Task")
    ax.set_xticks(range(len(checkpoints)), [CHECKPOINT_NAMES[c] for c in checkpoints], rotation=15, ha="right")
    ax.set_yticks(range(len(tasks)), tasks)
    ax.tick_params(axis="y", labelsize=8)
    for i, task in enumerate(tasks):
        for j, ck in enumerate(checkpoints):
            value = values[(task, ck)]
            ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=7, color="white" if value >= 0.58 else INK)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Success rate")
    save(fig, "per-task-checkpoint-heatmap")


def main() -> None:
    style()
    checkpoint_figure()
    inference_figure()
    training_figure()
    heatmap_figure()
    print("generated checkpoint, inference, training, and task figures")


if __name__ == "__main__":
    main()
