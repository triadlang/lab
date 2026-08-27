#!/usr/bin/env python3
"""Observer/observed test with chemistry, multi-scale memory and a life filter.

The control already contains the eight-channel chemical layer.  The comparison
adds four continuous memory horizons and one personal filter vector per entity.
Every entity follows the same law; only its experienced stream and relational
history differ.  Nothing is reset, selected, clipped, or assigned a fixed life.

English version of ../../entre/simulate_filtro_vida_entre.py (same dynamics, translated
labels, English JSON keys and output filenames).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import combinations
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

DT, STEPS, SAVE_EVERY, FRAME_EVERY = 0.02, 6000, 10, 60
MEMORY_RATE, MEMORY_FEEDBACK = 0.24, 0.34
CHEM_DECAY, CHEM_PHASE_GAIN, CHEM_EDGE_GAIN, CHEM_MEMORY_GAIN = 0.16, 0.18, 1.20, 0.35
LIFE_RATE, LIFE_PHASE_GAIN, LIFE_CHEM_GAIN, EXPERIENCE_GAIN = 0.045, 0.08, 0.12, 0.07
MEMORY_RATES = np.array((0.025, 0.075, 0.22, 0.65))
MEMORY_WEIGHTS = 1/np.sqrt(np.arange(1, len(MEMORY_RATES)+1))
MEMORY_WEIGHTS /= MEMORY_WEIGHTS.sum()

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
    Scenario("one_way", "1. S₁ → O₁", ("S1", "O1"), (Edge("S1", "O1", .28),)),
    Scenario("mutual", "2. S₁ ↔ O₁", ("S1", "O1"),
             (Edge("S1", "O1", .28), Edge("O1", "S1", .28))),
    Scenario("nested", "3. O₂ ↔ (S₁ ↔ O₁)", ("S1", "O1", "O2"),
             (Edge("S1", "O1", .28), Edge("O1", "S1", .28), Edge("S1", "O2", .10),
              Edge("O2", "S1", .10), Edge("O1", "O2", .18), Edge("O2", "O1", .18))),
    Scenario("paired", "4. (S₁↔O₁) ↔ (S₂↔O₂)", ("S1", "O1", "S2", "O2"),
             (Edge("S1", "O1", .28), Edge("O1", "S1", .28), Edge("S2", "O2", .28),
              Edge("O2", "S2", .28), Edge("O1", "S2", .14), Edge("S2", "O1", .14),
              Edge("O2", "S1", .14), Edge("S1", "O2", .14))),
)


def unit(values: np.ndarray) -> np.ndarray:
    return values/np.linalg.norm(values)


gold = (math.sqrt(5)-1)/2
ii, jj = np.meshgrid(np.arange(1, N_CHEM+1), np.arange(1, N_CHEM+1), indexing="ij")
CHEM_COUPLING = .11*np.sin(np.pi*gold*ii*(jj+1))/math.sqrt(N_CHEM)
np.fill_diagonal(CHEM_COUPLING, 0)
PHASE_VECTOR = unit(np.cos(2*np.pi*gold*np.arange(1, N_CHEM+1)))
EDGE_VECTOR = unit(np.sin(2*np.pi*gold*np.arange(1, N_CHEM+1)))
MEMORY_VECTOR = unit(np.cos(2*np.pi*gold*(np.arange(1, N_CHEM+1)+.5)))
ALIGN_VECTOR = unit(np.sin(np.pi*gold*(np.arange(1, N_CHEM+1)+.25)))
MISMATCH_VECTOR = unit(np.cos(np.pi*gold*(np.arange(1, N_CHEM+1)+.75)))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1/(1+np.exp(-np.clip(x, -40, 40)))


def wrap(angle):
    return (angle+np.pi)%(2*np.pi)-np.pi


def initial_chem(scenario: Scenario) -> np.ndarray:
    result = np.empty((len(scenario.nodes), N_CHEM))
    for ni, name in enumerate(scenario.nodes):
        k = np.arange(1, N_CHEM+1)
        result[ni] = .14*np.sin(k*(PHASE0[name]+gold)+ni*gold)
    return result


def personal_stream(name: str, node_index: int, t: float) -> np.ndarray:
    k = np.arange(1, N_CHEM+1)
    signature = (node_index+1)*k*gold + PHASE0[name]
    return EXPERIENCE_GAIN*(
        np.sin((.07+.013*k)*t + signature)
        + .5*np.cos((.031+.009*k)*t + PHASE0[name]*k)
    )


def life_state(scenario: Scenario, theta: np.ndarray, edge_memory: np.ndarray,
               concentrations: np.ndarray, memory_bank: np.ndarray,
               life_filter: np.ndarray, t: float):
    idx = {name: i for i, name in enumerate(scenario.nodes)}
    experience = concentrations-.5
    for ni, name in enumerate(scenario.nodes):
        experience[ni] += personal_stream(name, ni, t)
    for ei, edge in enumerate(scenario.edges):
        src, dst = idx[edge.src], idx[edge.dst]
        delta = float(wrap(theta[src]-theta[dst]))
        contrast = concentrations[src]-concentrations[dst]
        experience[dst] += edge.strength*(
            .28*contrast + .18*edge_memory[ei]*ALIGN_VECTOR
            + .12*math.sin(delta)*MISMATCH_VECTOR
        )
        experience[src] += edge.strength*.05*math.cos(delta)*ALIGN_VECTOR
    memory_term = np.tensordot(memory_bank, MEMORY_WEIGHTS, axes=([1], [0]))
    filter_target = experience*memory_term + .35*(experience-memory_term)
    life_output = np.tanh(memory_term + life_filter*(experience+.5*memory_term))
    return experience, memory_term, filter_target, life_output


def derivative(scenario: Scenario, theta: np.ndarray, edge_memory: np.ndarray,
               latent: np.ndarray, memory_bank: np.ndarray, life_filter: np.ndarray,
               t: float, with_life: bool):
    idx = {name: i for i, name in enumerate(scenario.nodes)}
    concentrations = sigmoid(latent)
    dtheta = np.array([OMEGA[name]+.012*math.sin(.31*t+PHASE0[name]) for name in scenario.nodes])
    dedge = np.zeros(len(scenario.edges))
    dlatent = np.zeros_like(latent)
    dbank = np.zeros_like(memory_bank)
    dfilter = np.zeros_like(life_filter)
    experience, memory_term, filter_target, life_output = life_state(
        scenario, theta, edge_memory, concentrations, memory_bank, life_filter, t
    )

    for ni in range(len(scenario.nodes)):
        centered = concentrations[ni]-.5
        dlatent[ni] = -CHEM_DECAY*latent[ni] + CHEM_COUPLING@centered
        dtheta[ni] += CHEM_PHASE_GAIN*float(PHASE_VECTOR@centered)
        if with_life:
            dbank[ni] = MEMORY_RATES[:, None]*(experience[ni]-memory_bank[ni])
            dfilter[ni] = LIFE_RATE*(filter_target[ni]-life_filter[ni])
            dlatent[ni] += .16*memory_term[ni] + LIFE_CHEM_GAIN*life_output[ni]
            dtheta[ni] += LIFE_PHASE_GAIN*float(PHASE_VECTOR@life_output[ni])
            dtheta[ni] += .05*float(MEMORY_VECTOR@memory_term[ni])

    for ei, edge in enumerate(scenario.edges):
        src, dst = idx[edge.src], idx[edge.dst]
        delta = float(wrap(theta[src]-theta[dst]))
        pair = .5*(concentrations[src]+concentrations[dst])-.5
        contrast = concentrations[src]-concentrations[dst]
        mixed = float(EDGE_VECTOR@pair + .5*MEMORY_VECTOR@contrast)
        if with_life:
            mixed += .55*float(EDGE_VECTOR@(.5*(life_output[src]+life_output[dst])))
            mixed += .20*float(MEMORY_VECTOR@(memory_term[src]-memory_term[dst]))
        relation_gain = math.exp(CHEM_EDGE_GAIN*math.tanh(mixed))
        memory_rate = MEMORY_RATE*math.exp(CHEM_MEMORY_GAIN*math.tanh(float(MEMORY_VECTOR@pair)))
        dlatent[dst] += edge.strength*(
            .52*contrast + .18*edge_memory[ei]*ALIGN_VECTOR
            + .12*math.sin(delta)*MISMATCH_VECTOR
        )
        dlatent[src] += edge.strength*(-.11*contrast+.07*math.cos(delta)*ALIGN_VECTOR)
        dtheta[dst] += edge.strength*relation_gain*(
            math.sin(delta)+MEMORY_FEEDBACK*edge_memory[ei]*math.cos(delta)
        )
        dedge[ei] = memory_rate*(math.cos(delta)-edge_memory[ei])
    return dtheta, dedge, dlatent, dbank, dfilter, memory_term, life_output


def rk4_step(scenario, theta, edge_memory, latent, memory_bank, life_filter, t, with_life):
    state = (theta, edge_memory, latent, memory_bank, life_filter)
    k1 = derivative(scenario, *state, t, with_life)[:5]
    s2 = tuple(value+DT*change/2 for value, change in zip(state, k1))
    k2 = derivative(scenario, *s2, t+DT/2, with_life)[:5]
    s3 = tuple(value+DT*change/2 for value, change in zip(state, k2))
    k3 = derivative(scenario, *s3, t+DT/2, with_life)[:5]
    s4 = tuple(value+DT*change for value, change in zip(state, k3))
    k4 = derivative(scenario, *s4, t+DT, with_life)[:5]
    return tuple(value+DT*(a+2*b+2*c+d)/6 for value, a, b, c, d in zip(state, k1, k2, k3, k4))


def simulate(scenario: Scenario, with_life: bool):
    theta = np.array([PHASE0[name] for name in scenario.nodes])
    edge_memory = np.zeros(len(scenario.edges))
    latent = initial_chem(scenario)
    memory_bank = np.zeros((len(scenario.nodes), len(MEMORY_RATES), N_CHEM))
    life_filter = np.zeros((len(scenario.nodes), N_CHEM))
    saved = {key: [] for key in ("t", "theta", "edge", "velocity", "chem", "memory_term", "filter", "life")}
    for step in range(STEPS+1):
        t = step*DT
        if step%SAVE_EVERY == 0:
            deriv = derivative(scenario, theta, edge_memory, latent, memory_bank, life_filter, t, with_life)
            saved["t"].append(t); saved["theta"].append(theta.copy()); saved["edge"].append(edge_memory.copy())
            saved["velocity"].append(deriv[0].copy()); saved["chem"].append(sigmoid(latent.copy()))
            saved["memory_term"].append(deriv[5].copy()); saved["filter"].append(life_filter.copy())
            saved["life"].append(deriv[6].copy())
        if step<STEPS:
            theta, edge_memory, latent, memory_bank, life_filter = rk4_step(
                scenario, theta, edge_memory, latent, memory_bank, life_filter, t, with_life
            )
    return {key: np.asarray(value) for key, value in saved.items()}


def edge_index(scenario, src, dst):
    return next((i for i, edge in enumerate(scenario.edges) if edge.src==src and edge.dst==dst), None)


def add_metrics(scenario, data):
    idx = {name: i for i, name in enumerate(scenario.nodes)}
    qualities = []
    for ei, edge in enumerate(scenario.edges):
        src, dst = idx[edge.src], idx[edge.dst]
        delta = wrap(data["theta"][:, src]-data["theta"][:, dst])
        alignment = (1+np.cos(delta))/2
        continuity = 1/(1+np.abs(data["velocity"][:, src]-data["velocity"][:, dst]))
        qualities.append(np.clip(np.abs(data["edge"][:, ei])*alignment*continuity, 0, 1))
    quality = np.column_stack(qualities)
    reciprocal = all(edge_index(scenario, edge.dst, edge.src) is not None for edge in scenario.edges)
    data["closed"] = np.exp(np.mean(np.log(np.clip(quality, 1e-12, 1)), axis=1))*100 if reciprocal else np.zeros(len(data["t"]))
    data["chem_mean"] = data["chem"].mean(axis=1)
    divergences = []
    for snapshot in data["filter"]:
        pairs = [np.linalg.norm(snapshot[a]-snapshot[b])/math.sqrt(N_CHEM)*100
                 for a, b in combinations(range(len(scenario.nodes)), 2)]
        divergences.append(np.mean(pairs) if pairs else 0)
    data["filter_divergence"] = np.asarray(divergences)


results = {}
for scenario in SCENARIOS:
    control, life = simulate(scenario, False), simulate(scenario, True)
    add_metrics(scenario, control); add_metrics(scenario, life)
    i_s1 = scenario.nodes.index("S1")
    life["phase_shift"] = np.abs(wrap(life["theta"][:, i_s1]-control["theta"][:, i_s1]))/np.pi*100
    life["between_delta"] = life["closed"]-control["closed"]
    results[scenario.key] = {"control": control, "life": life}


sample_ids = np.arange(0, len(results["mutual"]["control"]["t"]), 25)
rows = []
for i in sample_ids:
    row = {"time": round(float(results["mutual"]["control"]["t"][i]), 1)}
    for key in ("one_way", "mutual", "nested", "paired"):
        life = results[key]["life"]
        row[f"{key}_phase"] = round(float(life["phase_shift"][i]), 3)
        row[f"{key}_between"] = round(float(life["between_delta"][i]), 3)
        row[f"{key}_filter"] = round(float(life["filter_divergence"][i]), 3)
    rows.append(row)
(OUT/"life-filter-between-data.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


BG, FG, MUTED, GRID = "#070914", "#edf1fb", "#929cb2", "#2a3145"
COLORS = {"one_way": "#62a8ff", "mutual": "#ffb84d", "nested": "#b985ff", "paired": "#5ed6a7"}
NODE_COLORS = {"S1": "#62a8ff", "O1": "#ffb84d", "S2": "#5ed6a7", "O2": "#b985ff"}


def positions(theta): return np.column_stack((np.cos(theta), np.sin(theta)))


def center_of(scenario, pos):
    idx = {name: i for i, name in enumerate(scenario.nodes)}
    p1 = (pos[idx["S1"]]+pos[idx["O1"]])/2
    if scenario.key in {"one_way", "mutual"}: return p1
    if scenario.key == "nested": return (p1+pos[idx["O2"]])/2
    return (p1+(pos[idx["S2"]]+pos[idx["O2"]])/2)/2


def arrow(ax, start, end, color, alpha, bend):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8, linewidth=.9,
                                color=color, alpha=alpha, connectionstyle=f"arc3,rad={bend:.2f}", shrinkA=7, shrinkB=7))


def draw_panel(ax, scenario, i, trail=0):
    control, life = results[scenario.key]["control"], results[scenario.key]["life"]
    ghost, solid = positions(control["theta"][i]), positions(life["theta"][i])
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color=GRID, lw=1))
    if trail:
        start=max(0, i-trail)
        for ni, name in enumerate(scenario.nodes):
            path=positions(life["theta"][start:i+1, ni]); ax.plot(path[:,0], path[:,1], color=NODE_COLORS[name], lw=.8, alpha=.30)
    counts={}
    for edge in scenario.edges:
        pair=tuple(sorted((edge.src,edge.dst))); counts[pair]=counts.get(pair,0)+1
    for ei, edge in enumerate(scenario.edges):
        src,dst=scenario.nodes.index(edge.src),scenario.nodes.index(edge.dst); pair=tuple(sorted((edge.src,edge.dst)))
        bend=(.12 if edge.src<edge.dst else -.12) if counts[pair]>1 else 0
        arrow(ax,solid[src],solid[dst],NODE_COLORS[edge.src],.18+.72*min(1,abs(life["edge"][i,ei])),bend)
    for ni,name in enumerate(scenario.nodes):
        ax.scatter(*ghost[ni],s=80,facecolors="none",edgecolors=MUTED,linewidths=1,alpha=.8,zorder=4)
        ax.scatter(*solid[ni],s=70,color=NODE_COLORS[name],edgecolors=BG,linewidths=.8,zorder=6)
        ax.text(*(solid[ni]*1.14),name.replace("1","₁").replace("2","₂"),color=FG,fontsize=8,ha="center",va="center")
    center=center_of(scenario,solid); profile=.5+.5*np.tanh(life["life"][i].mean(axis=0)*4)
    for k,value in enumerate(profile):
        angle=2*np.pi*k/N_CHEM; end=center+np.array((np.cos(angle),np.sin(angle)))*(.05+.18*value)
        ax.plot((center[0],end[0]),(center[1],end[1]),color=COLORS[scenario.key],lw=1.3,alpha=.30+.65*value,zorder=5)
    ax.text(-1.30,-1.19,f"chemical between {control['closed'][i]:.1f}% → life {life['closed'][i]:.1f}%",color=FG,fontsize=7)
    ax.text(-1.30,-1.33,f"Δ S₁ {life['phase_shift'][i]:.1f}% · filter difference {life['filter_divergence'][i]:.1f}",color=MUTED,fontsize=7)
    ax.set(xlim=(-1.38,1.38),ylim=(-1.43,1.34),aspect="equal"); ax.set_xticks([]);ax.set_yticks([]);ax.set_facecolor(BG)
    for spine in ax.spines.values(): spine.set_color(GRID)


def style(ax, signed=False):
    ax.set_facecolor(BG);ax.grid(color=GRID,alpha=.55,lw=.6);ax.tick_params(colors=MUTED,labelsize=8)
    for spine in ax.spines.values(): spine.set_color(GRID)
    ax.set_xlim(0,120)
    if signed:
        limit=max(5,max(np.max(np.abs(results[k]["life"]["between_delta"])) for k in ("mutual","nested","paired")))
        ax.set_ylim(-1.08*limit,1.08*limit);ax.axhline(0,color=FG,lw=.7,alpha=.45)
    else: ax.set_ylim(0,102)


fig=plt.figure(figsize=(15.5,11.2),dpi=150,facecolor=BG);gs=fig.add_gridspec(4,4,height_ratios=(1,1,1,.88),hspace=.20,wspace=.16)
fig.suptitle("Continuous memory and life filter: the same law, different experiences",color=FG,fontsize=17,y=.985)
for col,scenario in enumerate(SCENARIOS):
    for row,wanted in enumerate((0,45,120)):
        i=int(np.argmin(np.abs(results[scenario.key]["life"]["t"]-wanted)));ax=fig.add_subplot(gs[row,col]);draw_panel(ax,scenario,i,28 if row else 0)
        if row==0:ax.set_title(scenario.title,color=FG,fontsize=10,pad=7)
        ax.text(-1.30,1.20,f"t={wanted}",color=MUTED,fontsize=7)
ax1=fig.add_subplot(gs[3,:2]);ax2=fig.add_subplot(gs[3,2:])
for key in ("one_way","mutual","nested","paired"):
    life=results[key]["life"];title=next(s.title for s in SCENARIOS if s.key==key)
    ax1.plot(life["t"],life["phase_shift"],color=COLORS[key],lw=1.8,label=title)
    ax2.plot(life["t"],life["filter_divergence"],color=COLORS[key],lw=1.8,label=title)
style(ax1);style(ax2);ax2.set_ylim(0,2)
ax1.set_title("How much different histories shift S₁",color=FG,fontsize=11);ax2.set_title("How different the personal filters become",color=FG,fontsize=11)
ax1.set_xlabel("time",color=FG);ax2.set_xlabel("time",color=FG);ax1.set_ylabel("phase separation (%)",color=FG);ax2.set_ylabel("distance between filters",color=FG)
for ax in (ax1,ax2):ax.legend(frameon=False,labelcolor=FG,fontsize=7,loc="best")
fig.savefig(OUT/"memory-life-filter-change.png",facecolor=BG,bbox_inches="tight");plt.close(fig)


fig=plt.figure(figsize=(13.4,7.0),dpi=92,facecolor=BG);gs=fig.add_gridspec(2,4,height_ratios=(1,.72),hspace=.23,wspace=.20)
phase_axes=[fig.add_subplot(gs[0,i]) for i in range(4)];shift_ax=fig.add_subplot(gs[1,:2]);filter_ax=fig.add_subplot(gs[1,2:])
fig.suptitle("Life filter — outline: chemistry only · dot: chemistry + personal memory",color=FG,fontsize=14,y=.98)
frame_ids=np.arange(0,len(results["mutual"]["life"]["t"]),FRAME_EVERY//SAVE_EVERY)


def animate(frame):
    i=int(frame_ids[frame])
    for ax,scenario in zip(phase_axes,SCENARIOS):ax.clear();draw_panel(ax,scenario,i,34);ax.set_title(scenario.title,color=FG,fontsize=8,pad=5)
    shift_ax.clear();filter_ax.clear()
    for key in ("one_way","mutual","nested","paired"):
        life=results[key]["life"];title=next(s.title for s in SCENARIOS if s.key==key)
        shift_ax.plot(life["t"][:i+1],life["phase_shift"][:i+1],color=COLORS[key],lw=1.7,label=title)
        filter_ax.plot(life["t"][:i+1],life["filter_divergence"][:i+1],color=COLORS[key],lw=1.7,label=title)
    style(shift_ax);style(filter_ax);filter_ax.set_ylim(0,2);shift_ax.set_title("S₁ shift",color=FG,fontsize=9);filter_ax.set_title("difference between filters",color=FG,fontsize=9)
    shift_ax.set_xlabel("time",color=FG,fontsize=8);filter_ax.set_xlabel("time",color=FG,fontsize=8);shift_ax.set_ylabel("%",color=FG,fontsize=8)
    shift_ax.legend(frameon=False,labelcolor=FG,fontsize=6,loc="best");filter_ax.legend(frameon=False,labelcolor=FG,fontsize=6,loc="best");return []


anim=FuncAnimation(fig,animate,frames=len(frame_ids),interval=55,blit=False);anim.save(OUT/"memory-life-filter-change.gif",writer=PillowWriter(fps=18));plt.close(fig)


for scenario in SCENARIOS:
    control,life=results[scenario.key]["control"],results[scenario.key]["life"];tail=slice(len(life["t"])//2,None)
    print(scenario.key,f"phase_final={life['phase_shift'][-1]:.2f}",f"filter_final={life['filter_divergence'][-1]:.2f}",
          f"between_control={np.mean(control['closed'][tail]):.2f}",f"between_life={np.mean(life['closed'][tail]):.2f}",
          f"between_delta={np.mean(life['between_delta'][tail]):+.2f}")
