#!/usr/bin/env python3
"""Observer/observed role simulation with nested and paired relations.

Every entity is a continuous complex phase z=exp(i theta). Directed relations
carry their own continuous memory. No state is collapsed or reset.

Scenarios:
  1) one-way: S -> O
  2) mutual: S <-> O
  3) nested: O2 <-> (S <-> O1)
  4) paired: (S1 <-> O1) <-> (S2 <-> O2)

English version of ../../entre/simulate_observer_observed_relations.py (same dynamics,
translated labels, English JSON keys and output filenames).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts"
OUT.mkdir(exist_ok=True)

DT = 0.02
STEPS = 6000
T_END = STEPS * DT
SAVE_EVERY = 10
FRAME_EVERY = 60
MEMORY_RATE = 0.24
MEMORY_FEEDBACK = 0.34


OMEGA = {
    "S1": 1.000,
    "O1": 1.075,
    "S2": 0.965,
    "O2": 0.925,
}
PHASE0 = {
    "S1": 0.30,
    "O1": 2.40,
    "S2": -1.10,
    "O2": 1.05,
}


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    strength: float


@dataclass
class Scenario:
    key: str
    title: str
    nodes: list[str]
    edges: list[Edge]


SCENARIOS = [
    Scenario(
        "one_way",
        "1. S → O (no return)",
        ["S1", "O1"],
        [Edge("S1", "O1", 0.28)],
    ),
    Scenario(
        "mutual",
        "2. S ↔ O (mutual)",
        ["S1", "O1"],
        [Edge("S1", "O1", 0.28), Edge("O1", "S1", 0.28)],
    ),
    Scenario(
        "nested",
        "3. O₂ ↔ (S ↔ O₁)",
        ["S1", "O1", "O2"],
        [
            Edge("S1", "O1", 0.28), Edge("O1", "S1", 0.28),
            Edge("S1", "O2", 0.10), Edge("O2", "S1", 0.10),
            Edge("O1", "O2", 0.18), Edge("O2", "O1", 0.18),
        ],
    ),
    Scenario(
        "paired",
        "4. (S₁↔O₁) ↔ (S₂↔O₂)",
        ["S1", "O1", "S2", "O2"],
        [
            Edge("S1", "O1", 0.28), Edge("O1", "S1", 0.28),
            Edge("S2", "O2", 0.28), Edge("O2", "S2", 0.28),
            Edge("O1", "S2", 0.14), Edge("S2", "O1", 0.14),
            Edge("O2", "S1", 0.14), Edge("S1", "O2", 0.14),
        ],
    ),
]


def wrap(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def derivative(
    scenario: Scenario,
    theta: np.ndarray,
    memory: np.ndarray,
    t: float,
) -> tuple[np.ndarray, np.ndarray]:
    idx = {name: i for i, name in enumerate(scenario.nodes)}
    dtheta = np.array([OMEGA[name] for name in scenario.nodes], dtype=float)
    # Same tiny deterministic environmental ripple for a given named entity in
    # every scenario, so differences come from the relation graph.
    for i, name in enumerate(scenario.nodes):
        phase_tag = PHASE0[name]
        dtheta[i] += 0.012 * math.sin(0.31 * t + phase_tag)
    dmemory = np.zeros(len(scenario.edges), dtype=float)
    for ei, edge in enumerate(scenario.edges):
        src = idx[edge.src]
        dst = idx[edge.dst]
        delta = float(wrap(theta[src] - theta[dst]))
        dtheta[dst] += edge.strength * (
            math.sin(delta) + MEMORY_FEEDBACK * memory[ei] * math.cos(delta)
        )
        dmemory[ei] = MEMORY_RATE * (math.cos(delta) - memory[ei])
    return dtheta, dmemory


def rk4_step(
    scenario: Scenario,
    theta: np.ndarray,
    memory: np.ndarray,
    t: float,
) -> tuple[np.ndarray, np.ndarray]:
    k1t, k1m = derivative(scenario, theta, memory, t)
    k2t, k2m = derivative(scenario, theta + 0.5 * DT * k1t, memory + 0.5 * DT * k1m, t + 0.5 * DT)
    k3t, k3m = derivative(scenario, theta + 0.5 * DT * k2t, memory + 0.5 * DT * k2m, t + 0.5 * DT)
    k4t, k4m = derivative(scenario, theta + DT * k3t, memory + DT * k3m, t + DT)
    theta_next = theta + (DT / 6.0) * (k1t + 2 * k2t + 2 * k3t + k4t)
    memory_next = memory + (DT / 6.0) * (k1m + 2 * k2m + 2 * k3m + k4m)
    return theta_next, memory_next


def edge_index(scenario: Scenario, src: str, dst: str) -> int | None:
    for i, edge in enumerate(scenario.edges):
        if edge.src == src and edge.dst == dst:
            return i
    return None


results: dict[str, dict[str, np.ndarray | Scenario]] = {}
for scenario in SCENARIOS:
    theta = np.array([PHASE0[name] for name in scenario.nodes], dtype=float)
    memory = np.zeros(len(scenario.edges), dtype=float)
    saved_t = []
    saved_theta = []
    saved_memory = []
    saved_velocity = []
    for step in range(STEPS + 1):
        t = step * DT
        if step % SAVE_EVERY == 0:
            vel, _ = derivative(scenario, theta, memory, t)
            saved_t.append(t)
            saved_theta.append(theta.copy())
            saved_memory.append(memory.copy())
            saved_velocity.append(vel.copy())
        if step < STEPS:
            theta, memory = rk4_step(scenario, theta, memory, t)
    results[scenario.key] = {
        "scenario": scenario,
        "t": np.asarray(saved_t),
        "theta": np.asarray(saved_theta),
        "memory": np.asarray(saved_memory),
        "velocity": np.asarray(saved_velocity),
    }


baseline_theta = results["one_way"]["theta"][:, 0]
for scenario in SCENARIOS:
    data = results[scenario.key]
    theta = data["theta"]
    velocity = data["velocity"]
    memory = data["memory"]
    idx = {name: i for i, name in enumerate(scenario.nodes)}
    primary_gap = np.abs(wrap(theta[:, idx["O1"]] - theta[:, idx["S1"]])) / np.pi * 100.0
    pair_coherence = np.abs(
        0.5 * (np.exp(1j * theta[:, idx["S1"]]) + np.exp(1j * theta[:, idx["O1"]]))
    ) * 100.0
    network_coherence = np.abs(np.mean(np.exp(1j * theta), axis=1)) * 100.0
    backreaction = np.abs(wrap(theta[:, idx["S1"]] - baseline_theta)) / np.pi * 100.0
    velocity_gap = np.abs(velocity[:, idx["O1"]] - velocity[:, idx["S1"]])
    ei = edge_index(scenario, "S1", "O1")
    record = np.abs(memory[:, ei]) * 100.0 if ei is not None else np.zeros(len(theta))
    data.update({
        "gap": primary_gap,
        "pair_coherence": pair_coherence,
        "network_coherence": network_coherence,
        "backreaction": backreaction,
        "velocity_gap": velocity_gap,
        "record": record,
    })


BG = "#070914"
FG = "#edf1fb"
MUTED = "#929cb2"
GRID = "#2a3145"
SCENARIO_COLORS = {
    "one_way": "#62a8ff",
    "mutual": "#ffb84d",
    "nested": "#b985ff",
    "paired": "#5ed6a7",
}
NODE_COLORS = {
    "S1": "#62a8ff",
    "O1": "#ffb84d",
    "S2": "#5ed6a7",
    "O2": "#b985ff",
}


def phase_positions(theta: np.ndarray, radius: float = 1.0) -> np.ndarray:
    return radius * np.column_stack((np.cos(theta), np.sin(theta)))


def draw_arrow(ax: plt.Axes, start: np.ndarray, end: np.ndarray, color: str, alpha: float, bend: float) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=0.9,
        color=color,
        alpha=alpha,
        connectionstyle="arc3,rad=%.2f" % bend,
        shrinkA=7,
        shrinkB=7,
    )
    ax.add_patch(patch)


def draw_phase_panel(ax: plt.Axes, scenario: Scenario, sample_i: int, trail: int = 0) -> None:
    data = results[scenario.key]
    theta = data["theta"][sample_i]
    memory = data["memory"][sample_i]
    pos = phase_positions(theta)
    circle = plt.Circle((0, 0), 1.0, fill=False, color=GRID, lw=1.0)
    ax.add_patch(circle)
    if trail > 0:
        start = max(0, sample_i - trail)
        for ni, name in enumerate(scenario.nodes):
            path = phase_positions(data["theta"][start:sample_i + 1, ni])
            ax.plot(path[:, 0], path[:, 1], color=NODE_COLORS[name], lw=0.8, alpha=0.38)
    pair_counts: dict[tuple[str, str], int] = {}
    for edge in scenario.edges:
        unordered = tuple(sorted((edge.src, edge.dst)))
        pair_counts[unordered] = pair_counts.get(unordered, 0) + 1
    for ei, edge in enumerate(scenario.edges):
        src = scenario.nodes.index(edge.src)
        dst = scenario.nodes.index(edge.dst)
        unordered = tuple(sorted((edge.src, edge.dst)))
        bend = 0.0
        if pair_counts[unordered] > 1:
            bend = 0.13 if edge.src < edge.dst else -0.13
        alpha = 0.20 + 0.70 * min(1.0, abs(memory[ei]))
        draw_arrow(ax, pos[src], pos[dst], NODE_COLORS[edge.src], alpha, bend)
    for ni, name in enumerate(scenario.nodes):
        ax.scatter(pos[ni, 0], pos[ni, 1], s=75, color=NODE_COLORS[name], edgecolors=BG, linewidths=0.8, zorder=5)
        ax.text(pos[ni, 0] * 1.14, pos[ni, 1] * 1.14, name.replace("1", "₁").replace("2", "₂"),
                color=FG, fontsize=8, ha="center", va="center")
    ax.text(-1.25, -1.24, "O₁–S₁ difference: %.1f%%" % data["gap"][sample_i], color=FG, fontsize=7)
    ax.text(-1.25, -1.38, "return on S₁: %.1f%%" % data["backreaction"][sample_i], color=MUTED, fontsize=7)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.45, 1.32)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def style_chart(ax: plt.Axes) -> None:
    ax.set_facecolor(BG)
    ax.grid(color=GRID, alpha=0.55, lw=0.7)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def make_static() -> None:
    fig = plt.figure(figsize=(17, 11), facecolor=BG)
    gs = fig.add_gridspec(4, 4, height_ratios=(1, 1, 1, 0.74), hspace=0.20, wspace=0.08)
    sample_times = [0.0, 45.0, 120.0]
    t_axis = results["one_way"]["t"]
    for row, time_value in enumerate(sample_times):
        sample_i = int(np.argmin(np.abs(t_axis - time_value)))
        for col, scenario in enumerate(SCENARIOS):
            ax = fig.add_subplot(gs[row, col])
            draw_phase_panel(ax, scenario, sample_i)
            if row == 0:
                ax.set_title(scenario.title, color=FG, fontsize=10, pad=6)
            ax.text(0.03, 0.96, "t=%.0f" % t_axis[sample_i], transform=ax.transAxes, color=FG, fontsize=7, ha="left", va="top")

    sub = gs[3, :].subgridspec(1, 2, wspace=0.18)
    ax = fig.add_subplot(sub[0, 0])
    for scenario in SCENARIOS:
        ax.plot(results[scenario.key]["t"], results[scenario.key]["gap"], color=SCENARIO_COLORS[scenario.key], lw=1.8, label=scenario.title)
    ax.set_title("Instantaneous difference between observed and observer", color=FG, fontsize=11)
    ax.set_xlabel("time", color=FG)
    ax.set_ylabel("phase separation (%)", color=FG)
    ax.set_ylim(0, 105)
    style_chart(ax)
    ax.legend(frameon=False, labelcolor=FG, fontsize=7, ncol=2)

    ax = fig.add_subplot(sub[0, 1])
    for scenario in SCENARIOS[1:]:
        ax.plot(results[scenario.key]["t"], results[scenario.key]["backreaction"], color=SCENARIO_COLORS[scenario.key], lw=1.8, label=scenario.title)
    ax.set_title("How much observing changes the observed itself", color=FG, fontsize=11)
    ax.set_xlabel("time", color=FG)
    ax.set_ylabel("S₁ deviation vs. the no-return case (%)", color=FG)
    ax.set_ylim(0, 105)
    style_chart(ax)
    ax.legend(frameon=False, labelcolor=FG, fontsize=7)

    fig.suptitle("Observer and observed — one relation, one mutual relation, one observed relation and two coupled relations",
                 color=FG, fontsize=15, y=0.985)
    fig.savefig(OUT / "observer-observed-relations.png", dpi=170, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def make_animation() -> None:
    fig = plt.figure(figsize=(13.4, 7.0), facecolor=BG)
    gs = fig.add_gridspec(2, 4, height_ratios=(1.0, 0.42), hspace=0.22, wspace=0.08)
    phase_axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    ax_gap = fig.add_subplot(gs[1, :2])
    ax_back = fig.add_subplot(gs[1, 2:])
    sample_indices = np.arange(0, len(results["one_way"]["t"]), max(1, FRAME_EVERY // SAVE_EVERY))
    if sample_indices[-1] != len(results["one_way"]["t"]) - 1:
        sample_indices = np.append(sample_indices, len(results["one_way"]["t"]) - 1)

    def draw(frame_number: int):
        sample_i = int(sample_indices[frame_number])
        t_value = results["one_way"]["t"][sample_i]
        for ax, scenario in zip(phase_axes, SCENARIOS):
            ax.clear()
            draw_phase_panel(ax, scenario, sample_i, trail=24)
            ax.set_title(scenario.title, color=FG, fontsize=9, pad=5)
        ax_gap.clear()
        ax_back.clear()
        for scenario in SCENARIOS:
            ax_gap.plot(results[scenario.key]["t"][:sample_i + 1], results[scenario.key]["gap"][:sample_i + 1],
                        color=SCENARIO_COLORS[scenario.key], lw=1.4, label=scenario.title)
        ax_gap.set_xlim(0, T_END)
        ax_gap.set_ylim(0, 105)
        ax_gap.set_title("O₁–S₁ difference", color=FG, fontsize=9)
        ax_gap.set_xlabel("time", color=FG, fontsize=8)
        style_chart(ax_gap)
        ax_gap.legend(frameon=False, labelcolor=FG, fontsize=6, ncol=2)
        for scenario in SCENARIOS[1:]:
            ax_back.plot(results[scenario.key]["t"][:sample_i + 1], results[scenario.key]["backreaction"][:sample_i + 1],
                         color=SCENARIO_COLORS[scenario.key], lw=1.4, label=scenario.title)
        ax_back.set_xlim(0, T_END)
        ax_back.set_ylim(0, 105)
        ax_back.set_title("Change induced on the observed", color=FG, fontsize=9)
        ax_back.set_xlabel("time", color=FG, fontsize=8)
        style_chart(ax_back)
        ax_back.legend(frameon=False, labelcolor=FG, fontsize=6)
        fig.suptitle("Observer ↔ observed — t=%.1f" % t_value, color=FG, fontsize=14, y=0.985)
        return []

    animation = FuncAnimation(fig, draw, frames=len(sample_indices), interval=70, blit=False)
    animation.save(OUT / "observer-observed-relations.gif", writer=PillowWriter(fps=14), dpi=92)
    plt.close(fig)


def write_data() -> None:
    sample_times = np.linspace(0, T_END, 25)
    rows = []
    t_axis = results["one_way"]["t"]
    for t_value in sample_times:
        i = int(np.argmin(np.abs(t_axis - t_value)))
        rows.append({
            "time": round(float(t_axis[i]), 1),
            "mutual": round(float(results["mutual"]["backreaction"][i]), 3),
            "nested": round(float(results["nested"]["backreaction"][i]), 3),
            "two_pairs": round(float(results["paired"]["backreaction"][i]), 3),
        })
    (OUT / "observer-observed-relations-data.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    arrays = {"t": results["one_way"]["t"]}
    for scenario in SCENARIOS:
        for metric in ("gap", "pair_coherence", "network_coherence", "backreaction", "record"):
            arrays[scenario.key + "_" + metric] = results[scenario.key][metric]
    np.savez_compressed(OUT / "observer-observed-relations.npz", **arrays)


if __name__ == "__main__":
    make_static()
    make_animation()
    write_data()
    for scenario in SCENARIOS:
        data = results[scenario.key]
        print(scenario.key, "gap_final=%.2f" % data["gap"][-1], "backreaction_max=%.2f" % data["backreaction"].max(),
              "record_final=%.2f" % data["record"][-1])
    print("outputs", OUT)
