#!/usr/bin/env python3
"""
Imagens das "cordas"
====================
Não roda a dinâmica — só lê o estado final salvo pelo bravais_puro_3d.py
(bravais_outputs_3d/final_state.npz) e desenha três tipos de corda:

  1. CORDAS DE DENSIDADE — os filamentos que a densidade alta forma no
     espaço real: voxels do topo da densidade agrupados em componentes
     conexos, cada um desenhado como um fio 3D com sua espinha.

  2. CORDAS DE FASE — as linhas onde a fase de Ψ dá uma volta completa
     (2π) ao redor: são curvas fechadas/abertas genuinamente
     unidimensionais dentro do volume. Requer psi_f no npz (as rodadas
     novas do bravais_puro_3d.py já salvam; rodadas antigas não têm, e
     nesse caso este bloco é pulado com aviso).

  3. CORDAS-NOTA — cada modo forte do espectro desenhado como uma corda
     vibrando na sua direção: comprimento de onda 2π/|k|, amplitude
     proporcional à intensidade do modo. É o "cardápio de notas" do
     instrumento, visualizado como cordas.

Uso:
  python3 bravais_cordas.py                       (usa final_state.npz)
  python3 bravais_cordas.py caminho/arquivo.npz   (outro estado salvo)
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.colors import LinearSegmentedColormap

path = sys.argv[1] if len(sys.argv) > 1 else "bravais_outputs_3d/final_state.npz"
if not os.path.exists(path):
    raise SystemExit("não achei %s — rode o bravais_puro_3d.py antes" % path)
dat = np.load(path)
rho = dat["rho_f"]
psi = dat["psi_f"] if "psi_f" in dat.files else None
N = rho.shape[0]
# L não é salvo no npz; assume a caixa padrão (ajuste aqui se mudou)
L = float(os.environ.get("L", 32.0))
dx = L / N
os.makedirs("bravais_outputs_3d", exist_ok=True)

cmap_life = LinearSegmentedColormap.from_list(
    "life", ["#0b132b", "#1c2541", "#3a506b", "#5bc0be", "#f4d35e", "#ee6c4d"]
)

try:
    from scipy.ndimage import label as cc_label
except ImportError:
    def cc_label(mask):
        """Rotulagem de componentes conexos (6-vizinhos), numpy/BFS puro."""
        lab = np.zeros(mask.shape, dtype=np.int32)
        cur = 0
        idxs = np.argwhere(mask)
        idx_set = set(map(tuple, idxs))
        for seed in map(tuple, idxs):
            if lab[seed] != 0:
                continue
            cur += 1
            stack = [seed]
            lab[seed] = cur
            while stack:
                i, j, k = stack.pop()
                for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                          (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                    nb = (i + d[0], j + d[1], k + d[2])
                    if nb in idx_set and lab[nb] == 0:
                        lab[nb] = cur
                        stack.append(nb)
        return lab, cur

def to_phys(ijk):
    return ijk * dx - L / 2

# ================================================================ 1) CORDAS DE DENSIDADE
thr = np.percentile(rho, 97.0)
mask = rho > thr
lab, ncomp = cc_label(mask)
sizes = np.bincount(lab.ravel())[1:]
order = np.argsort(sizes)[::-1]
keep = [i + 1 for i in order[:24] if sizes[i] >= 8]  # os 24 maiores fios

fig = plt.figure(figsize=(15, 6))
for pi, (elev, azim, lab_v) in enumerate(
    [(25, 45, "diagonal"), (25, -45, "diagonal oposta"), (90, -90, "de cima")], 1
):
    ax = fig.add_subplot(1, 3, pi, projection="3d")
    for ci, comp in enumerate(keep):
        pts = to_phys(np.argwhere(lab == comp).astype(float))
        col = cmap_life(0.15 + 0.8 * (ci / max(len(keep) - 1, 1)))
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=14, color=col,
                   alpha=0.75, linewidths=0)
        # espinha: caminho por vizinho-mais-próximo a partir de uma ponta
        if len(pts) >= 4:
            rest = list(range(len(pts)))
            start = int(np.argmax(np.linalg.norm(pts - pts.mean(0), axis=1)))
            pathi = [start]; rest.remove(start)
            while rest:
                last = pts[pathi[-1]]
                dd = np.linalg.norm(pts[rest] - last, axis=1)
                j = int(np.argmin(dd))
                if dd[j] > 3.0 * dx:
                    break
                pathi.append(rest.pop(j))
            sp = pts[pathi]
            ax.plot(sp[:, 0], sp[:, 1], sp[:, 2], color=col, lw=2.0, alpha=0.95)
    ax.set_xlim(-L / 2, L / 2); ax.set_ylim(-L / 2, L / 2); ax.set_zlim(-L / 2, L / 2)
    ax.set_title(lab_v, fontsize=10)
    ax.view_init(elev=elev, azim=azim)
    ax.tick_params(labelsize=6)
plt.suptitle("Cordas de densidade — filamentos do topo 3%% de |Ψ|² (%d fios)" % len(keep),
             fontsize=13)
plt.tight_layout()
plt.savefig("bravais_outputs_3d/cordas_densidade.png", dpi=140, bbox_inches="tight")
plt.close()
print("cordas_densidade.png  (%d fios)" % len(keep))

# ================================================================ 2) CORDAS DE FASE
if psi is not None:
    ph = np.angle(psi)

    def winding(a, b, c, d):
        """Voltas de fase ao redor de uma plaqueta (soma dos saltos)."""
        s = 0.0
        for u, v in ((a, b), (b, c), (c, d), (d, a)):
            dphi = v - u
            s += (dphi + np.pi) % (2 * np.pi) - np.pi
        return s

    # plaquetas nas 3 orientações, vetorizado
    def vortex_points(ph):
        pts = []
        # orientação z (plaqueta no plano xy)
        a = ph[:-1, :-1, :]; b = ph[1:, :-1, :]; c = ph[1:, 1:, :]; d = ph[:-1, 1:, :]
        w = ((b - a + np.pi) % (2 * np.pi) - np.pi) + \
            ((c - b + np.pi) % (2 * np.pi) - np.pi) + \
            ((d - c + np.pi) % (2 * np.pi) - np.pi) + \
            ((a - d + np.pi) % (2 * np.pi) - np.pi)
        iz = np.argwhere(np.abs(w) > np.pi)
        for i, j, k in iz:
            pts.append((i + 0.5, j + 0.5, k, np.sign(w[i, j, k]), 2))
        # orientação y (plano xz)
        a = ph[:-1, :, :-1]; b = ph[1:, :, :-1]; c = ph[1:, :, 1:]; d = ph[:-1, :, 1:]
        w = ((b - a + np.pi) % (2 * np.pi) - np.pi) + \
            ((c - b + np.pi) % (2 * np.pi) - np.pi) + \
            ((d - c + np.pi) % (2 * np.pi) - np.pi) + \
            ((a - d + np.pi) % (2 * np.pi) - np.pi)
        iy = np.argwhere(np.abs(w) > np.pi)
        for i, j, k in iy:
            pts.append((i + 0.5, j, k + 0.5, np.sign(w[i, j, k]), 1))
        # orientação x (plano yz)
        a = ph[:, :-1, :-1]; b = ph[:, 1:, :-1]; c = ph[:, 1:, 1:]; d = ph[:, :-1, 1:]
        w = ((b - a + np.pi) % (2 * np.pi) - np.pi) + \
            ((c - b + np.pi) % (2 * np.pi) - np.pi) + \
            ((d - c + np.pi) % (2 * np.pi) - np.pi) + \
            ((a - d + np.pi) % (2 * np.pi) - np.pi)
        ix = np.argwhere(np.abs(w) > np.pi)
        for i, j, k in ix:
            pts.append((i, j + 0.5, k + 0.5, np.sign(w[i, j, k]), 0))
        return np.array(pts) if pts else np.zeros((0, 5))

    vp = vortex_points(ph)
    print("cordas de fase: %d atravessamentos de plaqueta" % len(vp))
    if len(vp):
        xyz = to_phys(vp[:, :3])
        sgn = vp[:, 3]
        fig = plt.figure(figsize=(15, 6))
        for pi, (elev, azim, lab_v) in enumerate(
            [(25, 45, "diagonal"), (25, -45, "diagonal oposta"), (0, -90, "de frente")], 1
        ):
            ax = fig.add_subplot(1, 3, pi, projection="3d")
            pos = sgn > 0
            ax.scatter(xyz[pos, 0], xyz[pos, 1], xyz[pos, 2], s=4,
                       color="#ee6c4d", alpha=0.6, linewidths=0, label="volta +")
            ax.scatter(xyz[~pos, 0], xyz[~pos, 1], xyz[~pos, 2], s=4,
                       color="#5bc0be", alpha=0.6, linewidths=0, label="volta −")
            ax.set_xlim(-L / 2, L / 2); ax.set_ylim(-L / 2, L / 2)
            ax.set_zlim(-L / 2, L / 2)
            ax.set_title(lab_v, fontsize=10)
            ax.view_init(elev=elev, azim=azim)
            ax.tick_params(labelsize=6)
            if pi == 1:
                ax.legend(fontsize=8, loc="upper left")
        plt.suptitle(
            "Cordas de fase — linhas onde a fase de Ψ dá volta completa (%d pontos)" % len(vp),
            fontsize=13)
        plt.tight_layout()
        plt.savefig("bravais_outputs_3d/cordas_fase.png", dpi=140, bbox_inches="tight")
        plt.close()
        print("cordas_fase.png")
else:
    print("(cordas de fase puladas: o npz não tem psi_f — rode o "
          "bravais_puro_3d.py atualizado, que agora salva a fase)")

# ================================================================ 3) CORDAS-NOTA
fft3 = np.fft.fftshift(np.abs(np.fft.fftn(rho)))
power = fft3**2
power[N // 2, N // 2, N // 2] = 0.0
flat = np.argsort(power.ravel())[::-1]
n_notes = 40
seen = set()
notes = []
for fi in flat:
    ijk = np.unravel_index(fi, power.shape)
    kvec = (np.array(ijk) - N // 2) * (2 * np.pi / L)
    key = tuple(np.round(np.abs(kvec), 6))  # junta k e -k (mesma nota)
    if key in seen:
        continue
    seen.add(key)
    notes.append((kvec, power[ijk]))
    if len(notes) >= n_notes:
        break

fig = plt.figure(figsize=(13, 11))
ax = fig.add_subplot(111, projection="3d")
pmax = notes[0][1]
s = np.linspace(-L / 2, L / 2, 200)
for ni, (kvec, pw) in enumerate(notes):
    kn = np.linalg.norm(kvec)
    if kn < 1e-9:
        continue
    u = kvec / kn                       # direção de propagação da nota
    # base perpendicular pra corda "vibrar"
    ref = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    w1 = np.cross(u, ref); w1 /= np.linalg.norm(w1)
    amp = 1.8 * (pw / pmax) ** 0.5
    line = np.outer(s, u) + np.outer(amp * np.sin(kn * s), w1)
    inside = np.all(np.abs(line) <= L / 2, axis=1)
    col = cmap_life(0.15 + 0.8 * ni / max(n_notes - 1, 1))
    ax.plot(line[inside, 0], line[inside, 1], line[inside, 2],
            color=col, lw=1.0 + 2.5 * pw / pmax, alpha=0.8)
ax.set_xlim(-L / 2, L / 2); ax.set_ylim(-L / 2, L / 2); ax.set_zlim(-L / 2, L / 2)
ax.set_title("Cordas-nota — os %d modos mais fortes desenhados como cordas\n"
             "(comprimento de onda 2π/|k|, espessura/amplitude ∝ intensidade)" % n_notes)
plt.tight_layout()
plt.savefig("bravais_outputs_3d/cordas_notas.png", dpi=150, bbox_inches="tight")
plt.close()
print("cordas_notas.png  (%d notas)" % len(notes))

print("\nimagens em bravais_outputs_3d/: cordas_densidade.png, "
      "cordas_fase.png (se houver psi), cordas_notas.png")
