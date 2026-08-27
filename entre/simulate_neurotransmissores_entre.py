#!/usr/bin/env python3
"""Continuous coupled neurotransmitter layer over the observer/observed test.

Each scenario is simulated twice from identical initial phases: base dynamics
and the same dynamics with eight coupled chemical channels.  Channel names are
identifiers, not isolated one-function assumptions.  All channels interact,
travel through directed relations, and feed phase and edge memory continuously.
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
CHEM_DECAY = 0.16
CHEM_PHASE_GAIN = 0.18
CHEM_EDGE_GAIN = 1.20
CHEM_MEMORY_GAIN = 0.35

OMEGA = {"S1": 1.000, "O1": 1.075, "S2": 0.965, "O2": 0.925}
PHASE0 = {"S1": 0.30, "O1": 2.40, "S2": -1.10, "O2": 1.05}
TRANSMITTERS = ("Glu", "GABA", "DA", "5-HT", "NE", "ACh", "OXT", "END")
N_CHEM = len(TRANSMITTERS)


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
    Scenario("one_way", "1. S₁ → O₁", ("S1", "O1"), (Edge("S1", "O1", 0.28),)),
    Scenario("mutual", "2. S₁ ↔ O₁", ("S1", "O1"),
             (Edge("S1", "O1", 0.28), Edge("O1", "S1", 0.28))),
    Scenario("nested", "3. O₂ ↔ (S₁ ↔ O₁)", ("S1", "O1", "O2"),
             (Edge("S1", "O1", 0.28), Edge("O1", "S1", 0.28),
              Edge("S1", "O2", 0.10), Edge("O2", "S1", 0.10),
              Edge("O1", "O2", 0.18), Edge("O2", "O1", 0.18))),
    Scenario("paired", "4. (S₁↔O₁) ↔ (S₂↔O₂)", ("S1", "O1", "S2", "O2"),
             (Edge("S1", "O1", 0.28), Edge("O1", "S1", 0.28),
              Edge("S2", "O2", 0.28), Edge("O2", "S2", 0.28),
              Edge("O1", "S2", 0.14), Edge("S2", "O1", 0.14),
              Edge("O2", "S1", 0.14), Edge("S1", "O2", 0.14))),
)


def normalized(values: np.ndarray) -> np.ndarray:
    return values / np.linalg.norm(values)


# Deterministic, dense coupling: no transmitter is assigned a single role.
gold = (math.sqrt(5.0) - 1.0) / 2.0
ii, jj = np.meshgrid(np.arange(1, N_CHEM + 1), np.arange(1, N_CHEM + 1), indexing="ij")
CHEM_COUPLING = 0.11 * np.sin(np.pi * gold * ii * (jj + 1)) / math.sqrt(N_CHEM)
np.fill_diagonal(CHEM_COUPLING, 0.0)
PHASE_VECTOR = normalized(np.cos(2*np.pi*gold*np.arange(1, N_CHEM+1)))
EDGE_VECTOR = normalized(np.sin(2*np.pi*gold*np.arange(1, N_CHEM+1)))
MEMORY_VECTOR = normalized(np.cos(2*np.pi*gold*(np.arange(1, N_CHEM+1)+0.5)))
ALIGN_VECTOR = normalized(np.sin(np.pi*gold*(np.arange(1, N_CHEM+1)+0.25)))
MISMATCH_VECTOR = normalized(np.cos(np.pi*gold*(np.arange(1, N_CHEM+1)+0.75)))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def wrap(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2*np.pi) - np.pi


def initial_chem(scenario: Scenario) -> np.ndarray:
    result = np.empty((len(scenario.nodes), N_CHEM), dtype=float)
    for ni, name in enumerate(scenario.nodes):
        k = np.arange(1, N_CHEM+1)
        result[ni] = 0.14*np.sin(k*(PHASE0[name]+gold) + ni*gold)
    return result


def derivative(scenario: Scenario, theta: np.ndarray, memory: np.ndarray,
               latent: np.ndarray, t: float, chemistry: bool):
    idx = {name: i for i, name in enumerate(scenario.nodes)}
    dtheta = np.array([OMEGA[name] for name in scenario.nodes], dtype=float)
    dmemory = np.zeros(len(scenario.edges), dtype=float)
    dlatent = np.zeros_like(latent)
    concentrations = sigmoid(latent)

    for i, name in enumerate(scenario.nodes):
        dtheta[i] += 0.012*math.sin(0.31*t + PHASE0[name])
        if chemistry:
            centered = concentrations[i] - 0.5
            dlatent[i] = -CHEM_DECAY*latent[i] + CHEM_COUPLING @ centered
            dtheta[i] += CHEM_PHASE_GAIN*float(PHASE_VECTOR @ centered)

    for ei, edge in enumerate(scenario.edges):
        src, dst = idx[edge.src], idx[edge.dst]
        delta = float(wrap(theta[src]-theta[dst]))
        relation_gain = 1.0
        memory_rate = MEMORY_RATE
        if chemistry:
            pair = 0.5*(concentrations[src]+concentrations[dst]) - 0.5
            contrast = concentrations[src]-concentrations[dst]
            mixed = float(EDGE_VECTOR @ pair + 0.5*MEMORY_VECTOR @ contrast)
            relation_gain = math.exp(CHEM_EDGE_GAIN*math.tanh(mixed))
            memory_rate *= math.exp(CHEM_MEMORY_GAIN*math.tanh(float(MEMORY_VECTOR @ pair)))
            # Bidirectional chemical transport and relation-driven release.
            dlatent[dst] += edge.strength*(
                0.52*contrast + 0.18*memory[ei]*ALIGN_VECTOR
                + 0.12*math.sin(delta)*MISMATCH_VECTOR
            )
            dlatent[src] += edge.strength*(
                -0.11*contrast + 0.07*math.cos(delta)*ALIGN_VECTOR
            )

        dtheta[dst] += edge.strength*relation_gain*(
            math.sin(delta) + MEMORY_FEEDBACK*memory[ei]*math.cos(delta)
        )
        dmemory[ei] = memory_rate*(math.cos(delta)-memory[ei])
    return dtheta, dmemory, dlatent


def rk4_step(scenario: Scenario, theta: np.ndarray, memory: np.ndarray,
             latent: np.ndarray, t: float, chemistry: bool):
    k1 = derivative(scenario, theta, memory, latent, t, chemistry)
    k2 = derivative(scenario, theta+DT*k1[0]/2, memory+DT*k1[1]/2,
                    latent+DT*k1[2]/2, t+DT/2, chemistry)
    k3 = derivative(scenario, theta+DT*k2[0]/2, memory+DT*k2[1]/2,
                    latent+DT*k2[2]/2, t+DT/2, chemistry)
    k4 = derivative(scenario, theta+DT*k3[0], memory+DT*k3[1],
                    latent+DT*k3[2], t+DT, chemistry)
    return (
        theta + DT*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6,
        memory + DT*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6,
        latent + DT*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6,
    )


def simulate(scenario: Scenario, chemistry: bool):
    theta = np.array([PHASE0[name] for name in scenario.nodes], dtype=float)
    memory = np.zeros(len(scenario.edges), dtype=float)
    latent = initial_chem(scenario) if chemistry else np.zeros((len(scenario.nodes), N_CHEM))
    saved = {"t": [], "theta": [], "memory": [], "velocity": [], "chem": []}
    for step in range(STEPS+1):
        t = step*DT
        if step % SAVE_EVERY == 0:
            velocity, _, _ = derivative(scenario, theta, memory, latent, t, chemistry)
            saved["t"].append(t)
            saved["theta"].append(theta.copy())
            saved["memory"].append(memory.copy())
            saved["velocity"].append(velocity.copy())
            saved["chem"].append(sigmoid(latent.copy()))
        if step < STEPS:
            theta, memory, latent = rk4_step(scenario, theta, memory, latent, t, chemistry)
    return {key: np.asarray(value) for key, value in saved.items()}


def edge_index(scenario: Scenario, src: str, dst: str) -> int | None:
    return next((i for i, edge in enumerate(scenario.edges) if edge.src == src and edge.dst == dst), None)


def geometric_rows(values: np.ndarray) -> np.ndarray:
    return np.exp(np.mean(np.log(np.clip(values, 1e-12, 1.0)), axis=1))


def add_metrics(scenario: Scenario, data: dict[str, np.ndarray]):
    idx = {name: i for i, name in enumerate(scenario.nodes)}
    qualities = []
    for ei, edge in enumerate(scenario.edges):
        src, dst = idx[edge.src], idx[edge.dst]
        delta = wrap(data["theta"][:, src]-data["theta"][:, dst])
        alignment = (1+np.cos(delta))/2
        continuity = 1/(1+np.abs(data["velocity"][:, src]-data["velocity"][:, dst]))
        qualities.append(np.clip(np.abs(data["memory"][:, ei])*alignment*continuity, 0, 1))
    edge_quality = np.column_stack(qualities)
    reciprocal = all(edge_index(scenario, edge.dst, edge.src) is not None for edge in scenario.edges)
    data["closed_between"] = geometric_rows(edge_quality)*100 if reciprocal else np.zeros(len(data["t"]))
    data["trace"] = np.average(edge_quality, axis=1,
                               weights=np.array([edge.strength for edge in scenario.edges]))*100
    data["chem_mean"] = np.mean(data["chem"], axis=1)
    data["chem_spread"] = np.mean(np.std(data["chem"], axis=1), axis=1)*100


results: dict[str, dict[str, dict[str, np.ndarray]]] = {}
for scenario in SCENARIOS:
    base = simulate(scenario, False)
    chemical = simulate(scenario, True)
    add_metrics(scenario, base)
    add_metrics(scenario, chemical)
    i_s1 = scenario.nodes.index("S1")
    chemical["phase_shift"] = np.abs(wrap(chemical["theta"][:, i_s1]-base["theta"][:, i_s1]))/np.pi*100
    chemical["between_delta"] = chemical["closed_between"]-base["closed_between"]
    results[scenario.key] = {"base": base, "chemical": chemical}


sample_ids = np.arange(0, len(results["mutual"]["base"]["t"]), 25)
rows = []
for i in sample_ids:
    row = {"tempo": round(float(results["mutual"]["base"]["t"][i]), 1)}
    for key in ("mutual", "nested", "paired"):
        row[f"{key}_fase"] = round(float(results[key]["chemical"]["phase_shift"][i]), 3)
        row[f"{key}_entre"] = round(float(results[key]["chemical"]["between_delta"][i]), 3)
    rows.append(row)
(OUT/"neurotransmissores-entre-dados.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
)


BG, FG, MUTED, GRID = "#070914", "#edf1fb", "#929cb2", "#2a3145"
COLORS = {"one_way": "#62a8ff", "mutual": "#ffb84d", "nested": "#b985ff", "paired": "#5ed6a7"}
NODE_COLORS = {"S1": "#62a8ff", "O1": "#ffb84d", "S2": "#5ed6a7", "O2": "#b985ff"}
CHEM_COLORS = plt.cm.turbo(np.linspace(0.05, 0.95, N_CHEM))


def positions(theta: np.ndarray, radius: float = 1.0):
    return radius*np.column_stack((np.cos(theta), np.sin(theta)))


def relation_center(scenario: Scenario, pos: np.ndarray):
    idx = {name: i for i, name in enumerate(scenario.nodes)}
    p1 = (pos[idx["S1"]]+pos[idx["O1"]])/2
    if scenario.key in {"one_way", "mutual"}: return p1
    if scenario.key == "nested": return (p1+pos[idx["O2"]])/2
    p2 = (pos[idx["S2"]]+pos[idx["O2"]])/2
    return (p1+p2)/2


def draw_arrow(ax, start, end, color, alpha, bend):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8, linewidth=.9,
                                color=color, alpha=alpha, connectionstyle=f"arc3,rad={bend:.2f}",
                                shrinkA=7, shrinkB=7))


def draw_panel(ax, scenario: Scenario, sample_i: int, trail: int = 0):
    base, chem = results[scenario.key]["base"], results[scenario.key]["chemical"]
    base_pos, chem_pos = positions(base["theta"][sample_i]), positions(chem["theta"][sample_i])
    ax.add_patch(plt.Circle((0,0),1,fill=False,color=GRID,lw=1))
    if trail:
        start=max(0,sample_i-trail)
        for ni,name in enumerate(scenario.nodes):
            path=positions(chem["theta"][start:sample_i+1,ni])
            ax.plot(path[:,0],path[:,1],color=NODE_COLORS[name],lw=.8,alpha=.30)
    pair_counts={}
    for edge in scenario.edges:
        pair=tuple(sorted((edge.src,edge.dst))); pair_counts[pair]=pair_counts.get(pair,0)+1
    for ei,edge in enumerate(scenario.edges):
        src,dst=scenario.nodes.index(edge.src),scenario.nodes.index(edge.dst)
        pair=tuple(sorted((edge.src,edge.dst)))
        bend=(.12 if edge.src<edge.dst else -.12) if pair_counts[pair]>1 else 0
        draw_arrow(ax,chem_pos[src],chem_pos[dst],NODE_COLORS[edge.src],
                   .18+.72*min(1,abs(chem["memory"][sample_i,ei])),bend)

    # Ghost = original algorithm; solid = algorithm with chemical layer.
    for ni,name in enumerate(scenario.nodes):
        ax.scatter(*base_pos[ni],s=78,facecolors="none",edgecolors=MUTED,linewidths=1,alpha=.8,zorder=4)
        ax.scatter(*chem_pos[ni],s=68,color=NODE_COLORS[name],edgecolors=BG,linewidths=.8,zorder=6)
        label=name.replace("1","₁").replace("2","₂")
        ax.text(*(chem_pos[ni]*1.14),label,color=FG,fontsize=8,ha="center",va="center")

    center=relation_center(scenario,chem_pos)
    intensity=float(chem["closed_between"][sample_i])/100
    for size,alpha in ((720,.025),(390,.055),(170,.12)):
        ax.scatter(*center,s=size*(.15+.85*intensity),color=COLORS[scenario.key],
                   alpha=alpha+intensity*.11,edgecolors="none",zorder=2)
    # Eight spokes: current mean concentration of each chemical channel.
    mean_chem=chem["chem_mean"][sample_i]
    for k,value in enumerate(mean_chem):
        angle=2*np.pi*k/N_CHEM
        end=center+np.array([np.cos(angle),np.sin(angle)])*(.05+.18*value)
        ax.plot([center[0],end[0]],[center[1],end[1]],color=CHEM_COLORS[k],lw=1.3,alpha=.75,zorder=5)

    ax.text(-1.30,-1.19,f"entre base {base['closed_between'][sample_i]:.1f}% → químico {chem['closed_between'][sample_i]:.1f}%",
            color=FG,fontsize=7)
    ax.text(-1.30,-1.33,f"Δ entre {chem['between_delta'][sample_i]:+.1f} · deslocamento S₁ {chem['phase_shift'][sample_i]:.1f}%",
            color=MUTED,fontsize=7)
    ax.set(xlim=(-1.38,1.38),ylim=(-1.43,1.34),aspect="equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor(BG)
    for spine in ax.spines.values(): spine.set_color(GRID)


def style_chart(ax, signed=False):
    ax.set_facecolor(BG); ax.grid(color=GRID,alpha=.55,lw=.6)
    ax.tick_params(colors=MUTED,labelsize=8)
    for spine in ax.spines.values(): spine.set_color(GRID)
    ax.set_xlim(0,120)
    if signed:
        max_abs=max(5,max(np.max(np.abs(results[key]["chemical"]["between_delta"])) for key in ("mutual","nested","paired")))
        ax.set_ylim(-max_abs*1.08,max_abs*1.08)
        ax.axhline(0,color=FG,lw=.7,alpha=.45)
    else: ax.set_ylim(0,102)


fig=plt.figure(figsize=(15.5,11.2),dpi=150,facecolor=BG)
gs=fig.add_gridspec(4,4,height_ratios=[1,1,1,.88],hspace=.20,wspace=.16)
fig.suptitle("A mesma relação com e sem oito neurotransmissores contínuos e acoplados",
             color=FG,fontsize=17,y=.985)
for col,scenario in enumerate(SCENARIOS):
    for row,wanted in enumerate((0,45,120)):
        data=results[scenario.key]["base"]
        i=int(np.argmin(np.abs(data["t"]-wanted)))
        ax=fig.add_subplot(gs[row,col]); draw_panel(ax,scenario,i,trail=28 if row else 0)
        if row==0: ax.set_title(scenario.title,color=FG,fontsize=10,pad=7)
        ax.text(-1.30,1.20,f"t={wanted}",color=MUTED,fontsize=7)
ax1=fig.add_subplot(gs[3,:2]); ax2=fig.add_subplot(gs[3,2:])
for key in ("mutual","nested","paired"):
    chem=results[key]["chemical"]
    ax1.plot(chem["t"],chem["between_delta"],color=COLORS[key],lw=1.8,label=next(s.title for s in SCENARIOS if s.key==key))
    ax2.plot(chem["t"],chem["phase_shift"],color=COLORS[key],lw=1.8,label=next(s.title for s in SCENARIOS if s.key==key))
style_chart(ax1,signed=True); style_chart(ax2)
ax1.set_title("Mudança do estado do entre causada pela química",color=FG,fontsize=11)
ax1.set_xlabel("tempo",color=FG); ax1.set_ylabel("químico − base (p.p.)",color=FG)
ax2.set_title("Quanto a química desloca o próprio observado S₁",color=FG,fontsize=11)
ax2.set_xlabel("tempo",color=FG); ax2.set_ylabel("separação de fase (%)",color=FG)
for ax in (ax1,ax2): ax.legend(frameon=False,labelcolor=FG,fontsize=8,loc="best")
fig.savefig(OUT/"neurotransmissores-mudanca-no-entre.png",facecolor=BG,bbox_inches="tight")
plt.close(fig)


fig=plt.figure(figsize=(13.4,7.0),dpi=92,facecolor=BG)
gs=fig.add_gridspec(2,4,height_ratios=[1,.72],hspace=.23,wspace=.20)
phase_axes=[fig.add_subplot(gs[0,i]) for i in range(4)]
delta_ax=fig.add_subplot(gs[1,:2]); phase_ax=fig.add_subplot(gs[1,2:])
fig.suptitle("Neurotransmissores no entre — contorno: base · ponto: química · oito raios: canais",
             color=FG,fontsize=14,y=.98)
frame_ids=np.arange(0,len(results["mutual"]["base"]["t"]),FRAME_EVERY//SAVE_EVERY)


def animate(frame_number):
    i=int(frame_ids[frame_number])
    for ax,scenario in zip(phase_axes,SCENARIOS):
        ax.clear(); draw_panel(ax,scenario,i,trail=34); ax.set_title(scenario.title,color=FG,fontsize=8,pad=5)
    delta_ax.clear(); phase_ax.clear()
    for key in ("mutual","nested","paired"):
        chem=results[key]["chemical"]; title=next(s.title for s in SCENARIOS if s.key==key)
        delta_ax.plot(chem["t"][:i+1],chem["between_delta"][:i+1],color=COLORS[key],lw=1.7,label=title)
        phase_ax.plot(chem["t"][:i+1],chem["phase_shift"][:i+1],color=COLORS[key],lw=1.7,label=title)
    style_chart(delta_ax,signed=True); style_chart(phase_ax)
    delta_ax.set_title("Δ estado do entre",color=FG,fontsize=9); phase_ax.set_title("deslocamento de S₁",color=FG,fontsize=9)
    delta_ax.set_xlabel("tempo",color=FG,fontsize=8); phase_ax.set_xlabel("tempo",color=FG,fontsize=8)
    delta_ax.set_ylabel("p.p.",color=FG,fontsize=8); phase_ax.set_ylabel("%",color=FG,fontsize=8)
    delta_ax.legend(frameon=False,labelcolor=FG,fontsize=6,loc="best")
    phase_ax.legend(frameon=False,labelcolor=FG,fontsize=6,loc="best")
    return []


anim=FuncAnimation(fig,animate,frames=len(frame_ids),interval=55,blit=False)
anim.save(OUT/"neurotransmissores-mudanca-no-entre.gif",writer=PillowWriter(fps=18))
plt.close(fig)


for scenario in SCENARIOS:
    base,chem=results[scenario.key]["base"],results[scenario.key]["chemical"]
    tail=slice(len(base["t"])//2,None)
    print(scenario.key,
          f"base_tail={np.mean(base['closed_between'][tail]):.2f}",
          f"chem_tail={np.mean(chem['closed_between'][tail]):.2f}",
          f"delta_tail={np.mean(chem['between_delta'][tail]):+.2f}",
          f"phase_final={chem['phase_shift'][-1]:.2f}",
          "chem_final="+",".join(f"{name}:{value:.3f}" for name,value in zip(TRANSMITTERS,chem["chem_mean"][-1])))
