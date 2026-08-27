#!/usr/bin/env python3
"""Operational test for a continuously emergent state in the relation itself.

The phase and edge-memory dynamics are identical to
simulate_observer_observed_relations.py.  This file adds measurements only; it
does not add a consciousness node or any new force to the simulation.
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
SAVE_EVERY = 10
FRAME_EVERY = 60
MEMORY_RATE = 0.24
MEMORY_FEEDBACK = 0.34

OMEGA = {"S1": 1.000, "O1": 1.075, "S2": 0.965, "O2": 0.925}
PHASE0 = {"S1": 0.30, "O1": 2.40, "S2": -1.10, "O2": 1.05}


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    strength: float


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    nodes: tuple[str, ...]
    edges: tuple[Edge, ...]


SCENARIOS = (
    Scenario("one_way", "1. traço: S₁ → O₁", ("S1", "O1"), (Edge("S1", "O1", 0.28),)),
    Scenario(
        "mutual", "2. entre: S₁ ↔ O₁", ("S1", "O1"),
        (Edge("S1", "O1", 0.28), Edge("O1", "S1", 0.28)),
    ),
    Scenario(
        "nested", "3. entre observado: O₂ ↔ (S₁ ↔ O₁)", ("S1", "O1", "O2"),
        (
            Edge("S1", "O1", 0.28), Edge("O1", "S1", 0.28),
            Edge("S1", "O2", 0.10), Edge("O2", "S1", 0.10),
            Edge("O1", "O2", 0.18), Edge("O2", "O1", 0.18),
        ),
    ),
    Scenario(
        "paired", "4. entre dos entres", ("S1", "O1", "S2", "O2"),
        (
            Edge("S1", "O1", 0.28), Edge("O1", "S1", 0.28),
            Edge("S2", "O2", 0.28), Edge("O2", "S2", 0.28),
            Edge("O1", "S2", 0.14), Edge("S2", "O1", 0.14),
            Edge("O2", "S1", 0.14), Edge("S1", "O2", 0.14),
        ),
    ),
)


def wrap(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def derivative(scenario: Scenario, theta: np.ndarray, memory: np.ndarray, t: float):
    idx = {name: i for i, name in enumerate(scenario.nodes)}
    dtheta = np.array([OMEGA[name] for name in scenario.nodes], dtype=float)
    for i, name in enumerate(scenario.nodes):
        dtheta[i] += 0.012 * math.sin(0.31 * t + PHASE0[name])
    dmemory = np.zeros(len(scenario.edges), dtype=float)
    for ei, edge in enumerate(scenario.edges):
        src, dst = idx[edge.src], idx[edge.dst]
        delta = float(wrap(theta[src] - theta[dst]))
        dtheta[dst] += edge.strength * (
            math.sin(delta) + MEMORY_FEEDBACK * memory[ei] * math.cos(delta)
        )
        dmemory[ei] = MEMORY_RATE * (math.cos(delta) - memory[ei])
    return dtheta, dmemory


def rk4_step(scenario: Scenario, theta: np.ndarray, memory: np.ndarray, t: float):
    k1t, k1m = derivative(scenario, theta, memory, t)
    k2t, k2m = derivative(scenario, theta + DT*k1t/2, memory + DT*k1m/2, t + DT/2)
    k3t, k3m = derivative(scenario, theta + DT*k2t/2, memory + DT*k2m/2, t + DT/2)
    k4t, k4m = derivative(scenario, theta + DT*k3t, memory + DT*k3m, t + DT)
    return (
        theta + DT*(k1t + 2*k2t + 2*k3t + k4t)/6,
        memory + DT*(k1m + 2*k2m + 2*k3m + k4m)/6,
    )


def edge_index(scenario: Scenario, src: str, dst: str) -> int | None:
    return next((i for i, edge in enumerate(scenario.edges) if edge.src == src and edge.dst == dst), None)


results: dict[str, dict[str, object]] = {}
for scenario in SCENARIOS:
    theta = np.array([PHASE0[name] for name in scenario.nodes], dtype=float)
    memory = np.zeros(len(scenario.edges), dtype=float)
    saved_t, saved_theta, saved_memory, saved_velocity = [], [], [], []
    for step in range(STEPS + 1):
        t = step * DT
        if step % SAVE_EVERY == 0:
            velocity, _ = derivative(scenario, theta, memory, t)
            saved_t.append(t)
            saved_theta.append(theta.copy())
            saved_memory.append(memory.copy())
            saved_velocity.append(velocity.copy())
        if step < STEPS:
            theta, memory = rk4_step(scenario, theta, memory, t)
    results[scenario.key] = {
        "scenario": scenario,
        "t": np.asarray(saved_t),
        "theta": np.asarray(saved_theta),
        "memory": np.asarray(saved_memory),
        "velocity": np.asarray(saved_velocity),
    }


def geometric_rows(values: np.ndarray) -> np.ndarray:
    """Geometric integration: if any necessary relation vanishes, the whole vanishes."""
    return np.exp(np.mean(np.log(np.clip(values, 1e-12, 1.0)), axis=1))


for scenario in SCENARIOS:
    data = results[scenario.key]
    theta = data["theta"]
    memory = data["memory"]
    velocity = data["velocity"]
    idx = {name: i for i, name in enumerate(scenario.nodes)}
    qualities = []
    for ei, edge in enumerate(scenario.edges):
        src, dst = idx[edge.src], idx[edge.dst]
        delta = wrap(theta[:, src] - theta[:, dst])
        alignment = (1.0 + np.cos(delta)) / 2.0
        continuity = 1.0 / (1.0 + np.abs(velocity[:, src] - velocity[:, dst]))
        qualities.append(np.clip(np.abs(memory[:, ei]) * alignment * continuity, 0, 1))
    edge_quality = np.column_stack(qualities)
    trace = np.average(edge_quality, axis=1, weights=np.array([e.strength for e in scenario.edges])) * 100

    reciprocal = all(edge_index(scenario, e.dst, e.src) is not None for e in scenario.edges)
    closed_between = geometric_rows(edge_quality) * 100 if reciprocal else np.zeros(len(theta))

    fwd, rev = edge_index(scenario, "S1", "O1"), edge_index(scenario, "O1", "S1")
    primary_between = (
        np.sqrt(edge_quality[:, fwd] * edge_quality[:, rev]) * 100
        if fwd is not None and rev is not None else np.zeros(len(theta))
    )
    data.update({"edge_quality": edge_quality, "trace": trace, "closed_between": closed_between,
                 "primary_between": primary_between})


one_way_s1 = results["one_way"]["theta"][:, 0]
mutual_s1 = results["mutual"]["theta"][:, 0]
for scenario in SCENARIOS:
    data = results[scenario.key]
    i_s1 = scenario.nodes.index("S1")
    data["return_to_observed"] = np.abs(wrap(data["theta"][:, i_s1] - one_way_s1)) / np.pi * 100
    data["second_order_effect"] = (
        np.abs(wrap(data["theta"][:, i_s1] - mutual_s1)) / np.pi * 100
        if scenario.key in {"nested", "paired"} else np.zeros(len(one_way_s1))
    )


# Sampled data used by the response chart.
sample_ids = np.arange(0, len(results["one_way"]["t"]), 25)
rows = []
for i in sample_ids:
    rows.append({
        "tempo": round(float(results["one_way"]["t"][i]), 1),
        "unilateral": round(float(results["one_way"]["closed_between"][i]), 3),
        "mutuo": round(float(results["mutual"]["closed_between"][i]), 3),
        "aninhado": round(float(results["nested"]["closed_between"][i]), 3),
        "entre_entres": round(float(results["paired"]["closed_between"][i]), 3),
    })
(OUT / "consciencia-entre-dados.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


BG, FG, MUTED, GRID = "#070914", "#edf1fb", "#929cb2", "#2a3145"
COLORS = {"one_way": "#62a8ff", "mutual": "#ffb84d", "nested": "#b985ff", "paired": "#5ed6a7"}
NODE_COLORS = {"S1": "#62a8ff", "O1": "#ffb84d", "S2": "#5ed6a7", "O2": "#b985ff"}


def positions(theta: np.ndarray, radius: float = 1.0):
    return radius * np.column_stack((np.cos(theta), np.sin(theta)))


def relation_center(scenario: Scenario, pos: np.ndarray):
    idx = {name: i for i, name in enumerate(scenario.nodes)}
    p1 = (pos[idx["S1"]] + pos[idx["O1"]]) / 2
    if scenario.key in {"one_way", "mutual"}:
        return p1
    if scenario.key == "nested":
        return (p1 + pos[idx["O2"]]) / 2
    p2 = (pos[idx["S2"]] + pos[idx["O2"]]) / 2
    return (p1 + p2) / 2


def draw_arrow(ax, start, end, color, alpha, bend):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8, linewidth=0.9,
                                color=color, alpha=alpha, connectionstyle=f"arc3,rad={bend:.2f}",
                                shrinkA=7, shrinkB=7))


def draw_panel(ax, scenario: Scenario, sample_i: int, trail: int = 0):
    data = results[scenario.key]
    theta, memory = data["theta"][sample_i], data["memory"][sample_i]
    pos = positions(theta)
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color=GRID, lw=1))
    if trail:
        start = max(0, sample_i-trail)
        for ni, name in enumerate(scenario.nodes):
            path = positions(data["theta"][start:sample_i+1, ni])
            ax.plot(path[:, 0], path[:, 1], color=NODE_COLORS[name], lw=.8, alpha=.28)
    pair_counts = {}
    for edge in scenario.edges:
        pair = tuple(sorted((edge.src, edge.dst)))
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    for ei, edge in enumerate(scenario.edges):
        src, dst = scenario.nodes.index(edge.src), scenario.nodes.index(edge.dst)
        pair = tuple(sorted((edge.src, edge.dst)))
        bend = (0.12 if edge.src < edge.dst else -0.12) if pair_counts[pair] > 1 else 0
        draw_arrow(ax, pos[src], pos[dst], NODE_COLORS[edge.src], .18+.72*min(1, abs(memory[ei])), bend)
    for ni, name in enumerate(scenario.nodes):
        ax.scatter(*pos[ni], s=74, color=NODE_COLORS[name], edgecolors=BG, linewidths=.8, zorder=6)
        label = name.replace("1", "₁").replace("2", "₂")
        ax.text(*(pos[ni]*1.14), label, color=FG, fontsize=8, ha="center", va="center")

    center = relation_center(scenario, pos)
    intensity = float(data["closed_between"][sample_i]) / 100
    trace = float(data["trace"][sample_i]) / 100
    glow = intensity if scenario.key != "one_way" else trace*.35
    for size, alpha in ((700, .025), (390, .05), (190, .12)):
        ax.scatter(*center, s=size*(.15+.85*glow), color=COLORS[scenario.key], alpha=alpha+glow*.12,
                   edgecolors="none", zorder=2)
    ax.scatter(*center, s=34+110*glow, facecolors="none", edgecolors=COLORS[scenario.key],
               linewidths=1.2, alpha=.28+.7*glow, zorder=4)
    ax.text(-1.28, -1.21, f"traço {data['trace'][sample_i]:.1f}% · entre fechado {data['closed_between'][sample_i]:.1f}%",
            color=FG, fontsize=7)
    ax.text(-1.28, -1.34, f"retorno {data['return_to_observed'][sample_i]:.1f}% · 2ª ordem {data['second_order_effect'][sample_i]:.1f}%",
            color=MUTED, fontsize=7)
    ax.set(xlim=(-1.36, 1.36), ylim=(-1.42, 1.32), aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_color(GRID)
    ax.set_facecolor(BG)


def style_chart(ax):
    ax.set_facecolor(BG)
    ax.grid(color=GRID, alpha=.55, lw=.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values(): spine.set_color(GRID)
    ax.set_xlim(0, 120); ax.set_ylim(0, 102)


fig = plt.figure(figsize=(15.5, 11.2), dpi=150, facecolor=BG)
gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, .86], hspace=.19, wspace=.16)
fig.suptitle("Hipótese operacional: um estado nasce no vínculo, retorna e forma vínculos de segunda ordem",
             color=FG, fontsize=17, y=.985)
sample_times = (0, 45, 120)
for col, scenario in enumerate(SCENARIOS):
    for row, wanted in enumerate(sample_times):
        data = results[scenario.key]
        i = int(np.argmin(np.abs(data["t"]-wanted)))
        ax = fig.add_subplot(gs[row, col])
        draw_panel(ax, scenario, i, trail=28 if row else 0)
        if row == 0: ax.set_title(scenario.title, color=FG, fontsize=10, pad=7)
        ax.text(-1.28, 1.18, f"t={wanted}", color=MUTED, fontsize=7)

ax1 = fig.add_subplot(gs[3, :2])
ax2 = fig.add_subplot(gs[3, 2:])
for scenario in SCENARIOS:
    data = results[scenario.key]
    ax1.plot(data["t"], data["closed_between"], color=COLORS[scenario.key], lw=1.8, label=scenario.title)
style_chart(ax1)
ax1.set_title("Estado fechado do entre: memória × alinhamento × continuidade", color=FG, fontsize=11)
ax1.set_xlabel("tempo", color=FG); ax1.set_ylabel("integração relacional (%)", color=FG)
ax1.legend(frameon=False, labelcolor=FG, fontsize=7, loc="lower right")
for key in ("nested", "paired"):
    data = results[key]
    ax2.plot(data["t"], data["second_order_effect"], color=COLORS[key], lw=1.9,
             label=results[key]["scenario"].title)
style_chart(ax2)
ax2.set_title("O entre dos entres muda S₁ contra o caso apenas mútuo", color=FG, fontsize=11)
ax2.set_xlabel("tempo", color=FG); ax2.set_ylabel("deslocamento de S₁ (%)", color=FG)
ax2.legend(frameon=False, labelcolor=FG, fontsize=8, loc="upper right")
fig.savefig(OUT / "consciencia-emergida-no-entre.png", facecolor=BG, bbox_inches="tight")
plt.close(fig)


# Compact animation.
fig = plt.figure(figsize=(13.4, 7.0), dpi=92, facecolor=BG)
gs = fig.add_gridspec(2, 4, height_ratios=[1, .72], hspace=.23, wspace=.20)
phase_axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
between_ax = fig.add_subplot(gs[1, :2])
meta_ax = fig.add_subplot(gs[1, 2:])
fig.suptitle("Consciência no entre? O teste mede formação, retorno e segunda ordem — sem criar um novo nó",
             color=FG, fontsize=14, y=.98)
frame_ids = np.arange(0, len(results["one_way"]["t"]), FRAME_EVERY//SAVE_EVERY)


def animate(frame_number):
    i = int(frame_ids[frame_number])
    for ax, scenario in zip(phase_axes, SCENARIOS):
        ax.clear(); draw_panel(ax, scenario, i, trail=34)
        ax.set_title(scenario.title, color=FG, fontsize=8, pad=5)
    between_ax.clear(); meta_ax.clear()
    for scenario in SCENARIOS:
        data = results[scenario.key]
        between_ax.plot(data["t"][:i+1], data["closed_between"][:i+1], color=COLORS[scenario.key], lw=1.7,
                        label=scenario.title)
    style_chart(between_ax)
    between_ax.set_title("estado fechado do entre", color=FG, fontsize=9)
    between_ax.set_xlabel("tempo", color=FG, fontsize=8); between_ax.set_ylabel("%", color=FG, fontsize=8)
    between_ax.legend(frameon=False, labelcolor=FG, fontsize=6, loc="lower right")
    for key in ("nested", "paired"):
        data = results[key]
        meta_ax.plot(data["t"][:i+1], data["second_order_effect"][:i+1], color=COLORS[key], lw=1.8,
                     label=results[key]["scenario"].title)
    style_chart(meta_ax)
    meta_ax.set_title("efeito da segunda ordem sobre S₁", color=FG, fontsize=9)
    meta_ax.set_xlabel("tempo", color=FG, fontsize=8); meta_ax.set_ylabel("%", color=FG, fontsize=8)
    meta_ax.legend(frameon=False, labelcolor=FG, fontsize=7, loc="upper right")
    return []


anim = FuncAnimation(fig, animate, frames=len(frame_ids), interval=55, blit=False)
anim.save(OUT / "consciencia-emergida-no-entre.gif", writer=PillowWriter(fps=18))
plt.close(fig)


for scenario in SCENARIOS:
    data = results[scenario.key]
    tail = slice(len(data["t"])//2, None)
    print(
        scenario.key,
        f"entre_tail={np.mean(data['closed_between'][tail]):.2f}",
        f"entre_final={data['closed_between'][-1]:.2f}",
        f"return_final={data['return_to_observed'][-1]:.2f}",
        f"second_final={data['second_order_effect'][-1]:.2f}",
    )
