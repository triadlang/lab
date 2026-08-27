#!/usr/bin/env python3
"""
Images of the "strings"
=======================
Does not run the dynamics — it only reads the final state saved by
bravais_pure_3d.py (bravais_outputs_3d/final_state.npz) and draws three kinds
of string:

  1. DENSITY STRINGS — the filaments that the high density forms in real
     space: voxels from the top of the density grouped into connected
     components, each drawn as a 3D thread with its spine.

  2. PHASE STRINGS — the lines where the phase of Ψ makes a full turn (2π)
     around them: genuinely one-dimensional closed/open curves inside the
     volume. Requires psi_f in the npz (new runs of bravais_pure_3d.py
     already save it; old runs don't, in which case this block is skipped
     with a warning).

  3. NOTE STRINGS — each strong mode of the spectrum drawn as a string
     vibrating along its direction: wavelength 2π/|k|, amplitude
     proportional to the mode intensity. It is the instrument's "menu of
     notes", visualized as strings.

Usage:
  python3 bravais_strings.py                    (uses final_state.npz)
  python3 bravais_strings.py path/to/file.npz   (another saved state)

English version of ../../bravais/bravais_cordas.py (same processing, translated labels;
output images named strings_*.png).
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.colors import LinearSegmentedColormap

path = sys.argv[1] if len(sys.argv) > 1 else "bravais_outputs_3d/final_state.npz"
if not os.path.exists(path):
    raise SystemExit("could not find %s — run bravais_pure_3d.py first" % path)
dat = np.load(path)
rho = dat["rho_f"]
psi = dat["psi_f"] if "psi_f" in dat.files else None
N = rho.shape[0]
# L is not saved in the npz; assumes the default box (adjust here if changed)
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
        """Connected-component labeling (6-neighbors), pure numpy/BFS."""
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

# ================================================================ 1) DENSITY STRINGS
thr = np.percentile(rho, 97.0)
mask = rho > thr
lab, ncomp = cc_label(mask)
sizes = np.bincount(lab.ravel())[1:]
order = np.argsort(sizes)[::-1]
keep = [i + 1 for i in order[:24] if sizes[i] >= 8]  # the 24 largest threads

fig = plt.figure(figsize=(15, 6))
for pi, (elev, azim, lab_v) in enumerate(
    [(25, 45, "diagonal"), (25, -45, "opposite diagonal"), (90, -90, "from above")], 1
):
    ax = fig.add_subplot(1, 3, pi, projection="3d")
    for ci, comp in enumerate(keep):
        pts = to_phys(np.argwhere(lab == comp).astype(float))
        col = cmap_life(0.15 + 0.8 * (ci / max(len(keep) - 1, 1)))
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=14, color=col,
                   alpha=0.75, linewidths=0)
        # spine: nearest-neighbor path starting from one end
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
plt.suptitle("Density strings — filaments of the top 3%% of |Ψ|² (%d threads)" % len(keep),
             fontsize=13)
plt.tight_layout()
plt.savefig("bravais_outputs_3d/strings_density.png", dpi=140, bbox_inches="tight")
plt.close()
print("strings_density.png  (%d threads)" % len(keep))

# ================================================================ 2) PHASE STRINGS
if psi is not None:
    ph = np.angle(psi)

    def winding(a, b, c, d):
        """Phase turns around a plaquette (sum of the jumps)."""
        s = 0.0
        for u, v in ((a, b), (b, c), (c, d), (d, a)):
            dphi = v - u
            s += (dphi + np.pi) % (2 * np.pi) - np.pi
        return s

    # plaquettes in the 3 orientations, vectorized
    def vortex_points(ph):
        pts = []
        # z orientation (plaquette in the xy plane)
        a = ph[:-1, :-1, :]; b = ph[1:, :-1, :]; c = ph[1:, 1:, :]; d = ph[:-1, 1:, :]
        w = ((b - a + np.pi) % (2 * np.pi) - np.pi) + \
            ((c - b + np.pi) % (2 * np.pi) - np.pi) + \
            ((d - c + np.pi) % (2 * np.pi) - np.pi) + \
            ((a - d + np.pi) % (2 * np.pi) - np.pi)
        iz = np.argwhere(np.abs(w) > np.pi)
        for i, j, k in iz:
            pts.append((i + 0.5, j + 0.5, k, np.sign(w[i, j, k]), 2))
        # y orientation (xz plane)
        a = ph[:-1, :, :-1]; b = ph[1:, :, :-1]; c = ph[1:, :, 1:]; d = ph[:-1, :, 1:]
        w = ((b - a + np.pi) % (2 * np.pi) - np.pi) + \
            ((c - b + np.pi) % (2 * np.pi) - np.pi) + \
            ((d - c + np.pi) % (2 * np.pi) - np.pi) + \
            ((a - d + np.pi) % (2 * np.pi) - np.pi)
        iy = np.argwhere(np.abs(w) > np.pi)
        for i, j, k in iy:
            pts.append((i + 0.5, j, k + 0.5, np.sign(w[i, j, k]), 1))
        # x orientation (yz plane)
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
    print("phase strings: %d plaquette crossings" % len(vp))
    if len(vp):
        xyz = to_phys(vp[:, :3])
        sgn = vp[:, 3]
        fig = plt.figure(figsize=(15, 6))
        for pi, (elev, azim, lab_v) in enumerate(
            [(25, 45, "diagonal"), (25, -45, "opposite diagonal"), (0, -90, "from the front")], 1
        ):
            ax = fig.add_subplot(1, 3, pi, projection="3d")
            pos = sgn > 0
            ax.scatter(xyz[pos, 0], xyz[pos, 1], xyz[pos, 2], s=4,
                       color="#ee6c4d", alpha=0.6, linewidths=0, label="turn +")
            ax.scatter(xyz[~pos, 0], xyz[~pos, 1], xyz[~pos, 2], s=4,
                       color="#5bc0be", alpha=0.6, linewidths=0, label="turn −")
            ax.set_xlim(-L / 2, L / 2); ax.set_ylim(-L / 2, L / 2)
            ax.set_zlim(-L / 2, L / 2)
            ax.set_title(lab_v, fontsize=10)
            ax.view_init(elev=elev, azim=azim)
            ax.tick_params(labelsize=6)
            if pi == 1:
                ax.legend(fontsize=8, loc="upper left")
        plt.suptitle(
            "Phase strings — lines where the phase of Ψ makes a full turn (%d points)" % len(vp),
            fontsize=13)
        plt.tight_layout()
        plt.savefig("bravais_outputs_3d/strings_phase.png", dpi=140, bbox_inches="tight")
        plt.close()
        print("strings_phase.png")
else:
    print("(phase strings skipped: the npz has no psi_f — run the updated "
          "bravais_pure_3d.py, which now saves the phase)")

# ================================================================ 3) NOTE STRINGS
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
    key = tuple(np.round(np.abs(kvec), 6))  # merges k and -k (same note)
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
    u = kvec / kn                       # propagation direction of the note
    # perpendicular basis for the string to "vibrate"
    ref = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    w1 = np.cross(u, ref); w1 /= np.linalg.norm(w1)
    amp = 1.8 * (pw / pmax) ** 0.5
    line = np.outer(s, u) + np.outer(amp * np.sin(kn * s), w1)
    inside = np.all(np.abs(line) <= L / 2, axis=1)
    col = cmap_life(0.15 + 0.8 * ni / max(n_notes - 1, 1))
    ax.plot(line[inside, 0], line[inside, 1], line[inside, 2],
            color=col, lw=1.0 + 2.5 * pw / pmax, alpha=0.8)
ax.set_xlim(-L / 2, L / 2); ax.set_ylim(-L / 2, L / 2); ax.set_zlim(-L / 2, L / 2)
ax.set_title("Note strings — the %d strongest modes drawn as strings\n"
             "(wavelength 2π/|k|, thickness/amplitude ∝ intensity)" % n_notes)
plt.tight_layout()
plt.savefig("bravais_outputs_3d/strings_notes.png", dpi=150, bbox_inches="tight")
plt.close()
print("strings_notes.png  (%d notes)" % len(notes))

print("\nimages in bravais_outputs_3d/: strings_density.png, "
      "strings_phase.png (if psi is present), strings_notes.png")
