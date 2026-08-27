#!/usr/bin/env python3
"""
Bravais puro emergente — VERSÃO 3D
==================================
Regras rígidas (mesmas da versão 2D):
  - nenhum termo isolado
  - nenhuma calibração de coeficiente
  - nenhum forçar colapso / norma alvo
  - nenhum set_personality / set_lattice / set_Lambda
  - tudo sai de UMA proposta acoplada + ponto fixo implícito
  - normalização só se a norma explodir numericamente (proteção, não física)

Extensão fiel da versão restaurada (pré-split-step) para 3 dimensões:
  - grid X,Y,Z, FFT 3D (fftn), K² = kx²+ky²+kz²
  - V_mem ganha modos em z e acoplamentos cruzados 3D
  - v passa a 8 componentes (3 frequências + fases + cruzados)
  - picos detectados em 3D (maximum_filter 3D)
  - visualização: fatias centrais, scatter 3D dos picos com a1,a2,a3,
    isosuperfície (se scikit-image disponível), FFT (fatia central), séries temporais
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os

try:
    from scipy.ndimage import maximum_filter
except ImportError:
    def maximum_filter(arr, size=5):
        """Fallback numpy: filtro de máximo separável (bordas periódicas)."""
        r = size // 2
        out = arr
        for axis in range(arr.ndim):
            shifted = [np.roll(out, s, axis=axis) for s in range(-r, r + 1)]
            out = np.max(shifted, axis=0)
        return out

np.random.seed(None)
os.makedirs("bravais_outputs_3d", exist_ok=True)

# ---------------------------------------------------------------- grid 3D
L = 32.0
N = 64          # 64^3 = 262144 pontos; suba p/ 96 se tiver paciência
dx = L / N
x = np.linspace(-L / 2, L / 2, N, endpoint=False)
X, Y, Zg = np.meshgrid(x, x, x, indexing="ij")

k1 = np.fft.fftfreq(N, d=dx) * 2 * np.pi
KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing="ij")
K2 = KX**2 + KY**2 + KZ**2
Kabs = np.sqrt(K2 + 1e-30)

dV = dx**3


def multi_gaussian_field(n_centers=12):
    psi = np.zeros((N, N, N), dtype=np.complex128)
    centers = np.random.uniform(-0.3 * L, 0.3 * L, size=(n_centers, 3))
    widths = np.random.uniform(1.5, 3.5, size=n_centers)
    amps = np.random.uniform(0.3, 1.0, size=n_centers)
    for (cx, cy, cz), w, amp in zip(centers, widths, amps):
        phase = np.exp(1j * np.random.uniform(0, 2 * np.pi))
        psi += amp * phase * np.exp(
            -((X - cx) ** 2 + (Y - cy) ** 2 + (Zg - cz) ** 2) / (2 * w**2)
        )
    psi += 0.05 * (np.random.randn(N, N, N) + 1j * np.random.randn(N, N, N))
    return psi


psi0 = multi_gaussian_field(12)

Z = {
    "h": np.zeros(4),
    "b": np.zeros(4),
    "a": np.zeros(4),
    "n": np.zeros(4),
    "g": np.zeros(4),
    "ell": np.zeros(8),
    "v": np.zeros(8),   # 3D: mais graus de liberdade de rede
    "psi": psi0.copy(),
}

welford = {"mean": np.zeros(8), "M2": np.zeros(8), "count": 0}


def welford_update(measures):
    welford["count"] += 1
    c = welford["count"]
    delta = measures - welford["mean"]
    welford["mean"] += delta / c
    welford["M2"] += delta * (measures - welford["mean"])


def welford_z(measures):
    c = max(welford["count"], 1)
    var = welford["M2"] / max(c - 1, 1)
    return (measures - welford["mean"]) / (np.sqrt(var) + 1e-8)


def soft_bound(vec, cap=1.0):
    nrm = np.linalg.norm(vec)
    return vec / (1.0 + nrm / max(cap, 1e-12))


def project_state_vector(vec, radius=2.0):
    nrm = np.linalg.norm(vec)
    if nrm <= radius:
        return vec
    return vec * (radius / nrm)


guard_count = {"n": 0}


def psi_numeric_guard(psi):
    """Só age se a norma explodir ou colapsar numericamente."""
    nrm2 = np.sum(np.abs(psi) ** 2) * dV
    if np.isfinite(nrm2) and nrm2 > 100.0:
        guard_count["n"] += 1
    if not np.isfinite(nrm2) or nrm2 > 1e6:
        psi = np.nan_to_num(psi, nan=0.0, posinf=0.0, neginf=0.0)
        nrm2 = np.sum(np.abs(psi) ** 2) * dV
    if nrm2 < 1e-12:
        psi = psi + 1e-3 * (np.random.randn(N, N, N) + 1j * np.random.randn(N, N, N))
        nrm2 = np.sum(np.abs(psi) ** 2) * dV
    if nrm2 > 100.0:
        psi = psi * np.sqrt(50.0 / nrm2)
    return psi


def experience_signature(psi, Z):
    rho = np.abs(psi) ** 2
    mean_rho = np.mean(rho) + 1e-12
    std_rho = np.std(rho)
    fft = np.abs(np.fft.fftn(rho))
    total = np.sum(fft**2)
    order_energy = (total - fft[0, 0, 0] ** 2) / (total + 1e-12)
    gx = np.roll(rho, -1, 0) - np.roll(rho, 1, 0)
    gy = np.roll(rho, -1, 1) - np.roll(rho, 1, 1)
    gz = np.roll(rho, -1, 2) - np.roll(rho, 1, 2)
    rough = np.mean(gx**2 + gy**2 + gz**2)
    measures = np.array(
        [
            mean_rho,
            std_rho,
            order_energy,
            rough,
            np.linalg.norm(Z["ell"]),
            np.linalg.norm(Z["a"]),
            np.linalg.norm(Z["n"]),
            np.linalg.norm(Z["v"]),
        ]
    )
    welford_update(measures)
    z = welford_z(measures)
    chi = np.zeros(16)
    chi[:8] = z
    chi[8:12] = Z["ell"][:4]
    chi[12:16] = 0.5 * (Z["a"] + Z["n"])
    chi = chi / (np.sqrt(np.mean(chi**2)) + 1e-8)
    return chi, {
        "mean_rho": mean_rho,
        "std_rho": std_rho,
        "order": order_energy,
        "rough": rough,
    }


def coupled_proposal(chi, Z_mid):
    ell = Z_mid["ell"]
    a = Z_mid["a"]
    n = Z_mid["n"]
    g = Z_mid["g"]
    v = Z_mid["v"]

    drive = np.tanh(chi[:8] + 0.2 * ell)
    prop_h = drive[:4]
    prop_b = drive[4:8]
    prop_a = np.tanh(chi[:4] + 0.3 * a + 0.1 * n)
    prop_n = np.tanh(chi[4:8] + 0.3 * n + 0.1 * a)
    prop_g = np.tanh(chi[:4] + 0.25 * g + 0.1 * Z_mid["b"])
    prop_ell = np.tanh(chi[:8] + 0.2 * ell + 0.05 * np.concatenate([a, n]))
    prop_v = np.tanh(np.concatenate([chi[2:8], chi[0:2]]) + 0.15 * v)

    radial = np.exp(-(X**2 + Y**2 + Zg**2) / (0.25 * L**2 + 1e-8))
    e0, e1, e2, e3 = ell[0], ell[1], ell[2], ell[3]
    v0, v1, v2, v3, v4, v5, v6, v7 = v

    Lambda_scalar = np.tanh(e0 + 0.3 * chi[0] + 0.2 * a[0])
    alpha_scalar = 0.5 * (1.0 + np.tanh(e1 + 0.3 * chi[1]))
    Gamma_scalar = 0.5 * (1.0 + np.tanh(e2 + 0.3 * chi[2]))
    eta_scalar = 0.5 * (1.0 + np.tanh(e3 + 0.3 * chi[3]))
    sigma_frac = 1.0 + 0.5 * np.tanh(v0 + chi[4])

    # frequências emergentes nas 3 direções + modos cruzados
    fx = (2.0 * np.pi / L) * (3.0 + 2.0 * np.tanh(v0))
    fy = (2.0 * np.pi / L) * (3.0 + 2.0 * np.tanh(v1))
    fz = (2.0 * np.pi / L) * (3.0 + 2.0 * np.tanh(v2))
    fxy = (2.0 * np.pi / L) * (2.0 + 2.0 * np.tanh(v3))
    fyz = (2.0 * np.pi / L) * (2.0 + 2.0 * np.tanh(v4))
    fzx = (2.0 * np.pi / L) * (2.0 + 2.0 * np.tanh(v5))

    # λ_j >= 0: memória repulsiva (anti-colapso), conforme a base
    lam = lambda e: 0.5 * (1.0 + np.tanh(e))
    V_mem = (
        lam(e0) * radial
        + lam(e1) * np.sin(fx * X + np.tanh(v6))
        + lam(e2) * np.sin(fy * Y + np.tanh(v7))
        + lam(e3) * np.sin(fz * Zg + np.tanh(v6 - v7))
        + 0.5 * lam(ell[4]) * np.cos(fxy * (X + Y))
        + 0.5 * lam(ell[5]) * np.cos(fyz * (Y + Zg))
        + 0.5 * lam(ell[6]) * np.cos(fzx * (Zg + X))
        + 0.3 * lam(a[0]) * np.sin(fx * Y - fy * X)
        + 0.3 * lam(a[1]) * np.sin(fy * Zg - fz * Y)
    )
    V_scale = 0.5 * (1.0 + np.tanh(g[0] + chi[5]))
    V_mem = V_scale * V_mem
    V_ext = 0.15 * np.tanh(g[1]) * (
        (X / (L / 2)) ** 2 + (Y / (L / 2)) ** 2 + (Zg / (L / 2)) ** 2
    )

    return {
        "h": prop_h,
        "b": prop_b,
        "a": prop_a,
        "n": prop_n,
        "g": prop_g,
        "ell": prop_ell,
        "v": prop_v,
        "Lambda": Lambda_scalar,
        "alpha": alpha_scalar,
        "Gamma": Gamma_scalar,
        "eta": eta_scalar,
        "sigma": sigma_frac,
        "V_mem": V_mem,
        "V_ext": V_ext,
    }


def psi_flow(psi, prop, dt):
    rho = np.abs(psi) ** 2
    V = prop["V_ext"] + prop["Lambda"] * rho + prop["V_mem"]
    psi_k = np.fft.fftn(psi)
    kinetic = np.fft.ifftn(-0.5 * K2 * psi_k)
    frac = prop["alpha"] * np.fft.ifftn((Kabs ** prop["sigma"]) * psi_k)
    # -iΓ dentro do colchete de iħ∂tΨ  →  contribuição -Γψ em ∂tΨ (dissipação real)
    damp = -prop["Gamma"] * 0.05 * psi
    noise = prop["eta"] * 0.02 * (
        np.random.randn(N, N, N) + 1j * np.random.randn(N, N, N)
    )
    dpsi = -1j * (kinetic + V * psi + frac) + damp + noise
    return psi + dt * dpsi


def coupled_step(Z, dt=0.02, max_iter=5):
    Z0 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in Z.items()}
    Zcur = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in Z.items()}
    metrics = None
    prop = None
    for _ in range(max_iter):
        Zmid = {k: 0.5 * (Z0[k] + Zcur[k]) for k in Z0}
        chi, metrics = experience_signature(Zmid["psi"], Zmid)
        prop = coupled_proposal(chi, Zmid)
        keys = ["h", "b", "a", "n", "g", "ell", "v"]
        raw = {k: prop[k] - Zmid[k] for k in keys}
        stack = soft_bound(np.concatenate([raw[k] for k in keys]), cap=1.0)
        i0 = 0
        for k in keys:
            nsz = raw[k].size
            Zcur[k] = project_state_vector(Z0[k] + stack[i0 : i0 + nsz], radius=2.5)
            i0 += nsz
        psi_new = psi_flow(Zmid["psi"], prop, dt)
        Zcur["psi"] = psi_numeric_guard(psi_new)
        if np.linalg.norm(Zcur["ell"] - Zmid["ell"]) < 1e-3:
            break
    return Zcur, metrics, prop


# ---------------------------------------------------------------- evolução
STEPS = int(os.environ.get("STEPS", 1200))
dt = 0.025

print("Iniciando trajetória emergente 3D (versão restaurada pré-split-step)")
print("domínio: %.1f^3 | N=%d^3 | multi-gaussianos iniciais" % (L, N))

history = {
    "order": [], "mean_rho": [], "std_rho": [], "ell_norm": [],
    "v": [], "Lambda": [], "alpha": [], "Gamma": [], "norm": [],
}
snapshots = {}
snap_at = sorted({0, STEPS // 8, STEPS // 4, STEPS // 2, 3 * STEPS // 4, STEPS - 1})

for t in range(STEPS):
    Z, metrics, prop = coupled_step(Z, dt=dt)
    rho = np.abs(Z["psi"]) ** 2
    nrm = np.sqrt(np.sum(rho) * dV)
    history["order"].append(metrics["order"])
    history["mean_rho"].append(metrics["mean_rho"])
    history["std_rho"].append(metrics["std_rho"])
    history["ell_norm"].append(np.linalg.norm(Z["ell"]))
    history["v"].append(Z["v"].copy())
    history["Lambda"].append(prop["Lambda"])
    history["alpha"].append(prop["alpha"])
    history["Gamma"].append(prop["Gamma"])
    history["norm"].append(nrm)
    if t in snap_at:
        snapshots[t] = rho.copy()
    if t % 50 == 0:
        print(
            "t=%4d | order=%.4f | |ell|=%.3f | Λ=%.3f | ‖ψ‖=%.3f | ⟨ρ⟩=%.5f"
            % (t, metrics["order"], history["ell_norm"][-1], prop["Lambda"], nrm, metrics["mean_rho"])
        )

print("trajetória concluída.\n")

cmap_life = LinearSegmentedColormap.from_list(
    "life", ["#0b132b", "#1c2541", "#3a506b", "#5bc0be", "#f4d35e", "#ee6c4d"]
)
times = sorted(snapshots.keys())

# ------------------------------------------------ 1) evolução: fatias z=0
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for ax, t in zip(axes.flat, times):
    dens = snapshots[t][:, :, N // 2]
    ax.imshow(
        dens.T, extent=[-L / 2, L / 2, -L / 2, L / 2],
        origin="lower", cmap="magma", interpolation="bilinear",
    )
    ax.set_title("t = %d (fatia z=0)" % t)
    ax.set_xticks([]); ax.set_yticks([])
plt.suptitle("Densidade |Ψ|² 3D — fatias centrais", fontsize=13)
plt.tight_layout()
plt.savefig("bravais_outputs_3d/pure_evolution.png", dpi=150, bbox_inches="tight")
plt.close()

# ------------------------------------------------ 2) três planos ortogonais finais
rho_f = snapshots[times[-1]]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, (dens, lab) in zip(
    axes,
    [
        (rho_f[:, :, N // 2], "plano xy (z=0)"),
        (rho_f[:, N // 2, :], "plano xz (y=0)"),
        (rho_f[N // 2, :, :], "plano yz (x=0)"),
    ],
):
    im = ax.imshow(
        dens.T, extent=[-L / 2, L / 2, -L / 2, L / 2],
        origin="lower", cmap="magma", interpolation="bilinear",
    )
    ax.set_title(lab); ax.set_xticks([]); ax.set_yticks([])
plt.colorbar(im, ax=axes, fraction=0.02, label=r"$|\Psi|^2$")
plt.suptitle("Estado final — cortes ortogonais", fontsize=13)
plt.savefig("bravais_outputs_3d/pure_final_slices.png", dpi=150, bbox_inches="tight")
plt.close()

# ------------------------------------------------ 3) picos 3D + vetores de rede
thr = np.mean(rho_f) + 1.0 * np.std(rho_f)
local_max = (rho_f == maximum_filter(rho_f, size=5)) & (rho_f > thr)
peaks = np.argwhere(local_max)
pvals = rho_f[peaks[:, 0], peaks[:, 1], peaks[:, 2]] if len(peaks) else np.array([])

fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111, projection="3d")
if len(peaks):
    px = peaks[:, 0] * dx - L / 2
    py = peaks[:, 1] * dx - L / 2
    pz = peaks[:, 2] * dx - L / 2
    sizes = 40 + 160 * (pvals - pvals.min()) / (np.ptp(pvals) + 1e-12)
    sc = ax.scatter(px, py, pz, s=sizes, c=pvals, cmap=cmap_life,
                    edgecolors="white", linewidths=0.3, alpha=0.9, depthshade=True)
    fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.08, label=r"$|\Psi|^2$ no pico")

v = Z["v"]
scale = L / 8.0
a1 = scale * np.array([1.0 + 0.3 * np.tanh(v[0]), 0.3 * np.tanh(v[3]), 0.3 * np.tanh(v[5])])
a2 = scale * np.array([0.3 * np.tanh(v[3]), 1.0 + 0.3 * np.tanh(v[1]), 0.3 * np.tanh(v[4])])
a3 = scale * np.array([0.3 * np.tanh(v[5]), 0.3 * np.tanh(v[4]), 1.0 + 0.3 * np.tanh(v[2])])
for vec, col, lab in [(a1, "#5bc0be", "$a_1$"), (a2, "#f4d35e", "$a_2$"), (a3, "#ee6c4d", "$a_3$")]:
    ax.quiver(0, 0, 0, *vec, color=col, lw=2.5, arrow_length_ratio=0.12)
    ax.text(*(vec * 1.15), lab, color=col, fontsize=13)

ax.set_xlim(-L / 2, L / 2); ax.set_ylim(-L / 2, L / 2); ax.set_zlim(-L / 2, L / 2)
ax.set_title(
    "Rede emergente 3D | picos≈%d | |ℓ|=%.3f | Λ=%.3f"
    % (len(peaks), history["ell_norm"][-1], history["Lambda"][-1])
)
plt.tight_layout()
plt.savefig("bravais_outputs_3d/pure_final_peaks3d.png", dpi=160, bbox_inches="tight")
plt.close()

# ------------------------------------------------ 4) isosuperfície (opcional)
try:
    from skimage import measure

    iso = np.mean(rho_f) + 1.5 * np.std(rho_f)
    verts, faces, _, _ = measure.marching_cubes(rho_f, level=iso, spacing=(dx, dx, dx))
    verts -= L / 2
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(
        verts[:, 0], verts[:, 1], faces, verts[:, 2],
        cmap="magma", lw=0.1, alpha=0.85,
    )
    ax.set_xlim(-L / 2, L / 2); ax.set_ylim(-L / 2, L / 2); ax.set_zlim(-L / 2, L / 2)
    ax.set_title("Isosuperfície |Ψ|² = ⟨ρ⟩+1.5σ")
    plt.tight_layout()
    plt.savefig("bravais_outputs_3d/pure_isosurface.png", dpi=150, bbox_inches="tight")
    plt.close()
    has_iso = True
except Exception as e:
    has_iso = False
    print("(isosuperfície pulada: %s)" % e)

# ------------------------------------------------ 5) FFT (fatia central)
fft3 = np.fft.fftshift(np.abs(np.fft.fftn(rho_f)))
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, (sl, lab) in zip(
    axes,
    [
        (fft3[:, :, N // 2], "kx-ky"),
        (fft3[:, N // 2, :], "kx-kz"),
        (fft3[N // 2, :, :], "ky-kz"),
    ],
):
    ax.imshow(np.log1p(sl).T, origin="lower", cmap="viridis")
    ax.set_title("log|FFT(ρ)| — %s" % lab)
    ax.set_xticks([]); ax.set_yticks([])
plt.suptitle("Espectro 3D — periodicidade emergente (fatias)", fontsize=12)
plt.tight_layout()
plt.savefig("bravais_outputs_3d/pure_fft.png", dpi=140, bbox_inches="tight")
plt.close()

# ------------------------------------------------ 5a) espectro 3D visto de todos os lados
# mesmos números do fft3: pontos = top 1% da potência espectral (sem k=0),
# só a câmera muda entre os painéis.
power3 = fft3**2
power3[N // 2, N // 2, N // 2] = 0.0  # remove o pico k=0 (média), que ofusca o resto
thr_p = np.percentile(power3, 99.0)
kpts = np.argwhere(power3 > thr_p)
kvals = power3[kpts[:, 0], kpts[:, 1], kpts[:, 2]]
kc = (kpts - N // 2) * (2 * np.pi / L)  # coordenadas de k físicas, centradas

views = [
    (90, -90, "de cima (kx-ky)"),
    (0, -90, "de frente (kx-kz)"),
    (0, 0, "de lado (ky-kz)"),
    (0, 45, "lado diagonal 45°"),
    (35, 45, "diagonal alta 45°"),
    (35, 135, "diagonal alta 135°"),
    (35, 225, "diagonal alta 225°"),
    (35, 315, "diagonal alta 315°"),
    (-35, 45, "diagonal baixa 45°"),
]
fig = plt.figure(figsize=(15, 15))
for i, (elev, azim, lab) in enumerate(views, 1):
    ax = fig.add_subplot(3, 3, i, projection="3d")
    order_idx = np.argsort(kvals)
    ax.scatter(
        kc[order_idx, 0], kc[order_idx, 1], kc[order_idx, 2],
        c=np.log1p(kvals[order_idx]), cmap="viridis",
        s=6 + 40 * (kvals[order_idx] / kvals.max()),
        alpha=0.7, linewidths=0,
    )
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(lab, fontsize=10)
    kmax_plot = np.pi * N / L / 2
    ax.set_xlim(-kmax_plot, kmax_plot)
    ax.set_ylim(-kmax_plot, kmax_plot)
    ax.set_zlim(-kmax_plot, kmax_plot)
    ax.set_xlabel("kx", fontsize=7); ax.set_ylabel("ky", fontsize=7); ax.set_zlabel("kz", fontsize=7)
    ax.tick_params(labelsize=6)
plt.suptitle("Espectro 3D — top 1%% da potência, %d pontos, 9 ângulos de câmera" % len(kpts), fontsize=13)
plt.tight_layout()
plt.savefig("bravais_outputs_3d/pure_fft_views.png", dpi=130, bbox_inches="tight")
plt.close()

# ------------------------------------------------ 5a-2) teste da pirâmide
# hipótese: a intensidade log|FFT| tem forma de pirâmide (vista de cima = losango).
# teste 1: a fatia central como RELEVO (altura = intensidade), câmera girando.
kax = np.fft.fftshift(np.fft.fftfreq(N, d=dx)) * 2 * np.pi
KXs, KYs = np.meshgrid(kax, kax, indexing="ij")
relief = np.log1p(fft3[:, :, N // 2])
views_r = [
    (90, -90, "de cima"),
    (25, -90, "de frente"),
    (25, 0, "de lado"),
    (25, 45, "diagonal 45°"),
    (25, 135, "diagonal 135°"),
    (10, 45, "rasante 45°"),
]
fig = plt.figure(figsize=(15, 10))
for i, (elev, azim, lab) in enumerate(views_r, 1):
    ax = fig.add_subplot(2, 3, i, projection="3d")
    ax.plot_surface(KXs, KYs, relief, cmap="viridis", rstride=1, cstride=1,
                    linewidth=0, antialiased=True)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(lab, fontsize=10)
    ax.tick_params(labelsize=6)
plt.suptitle("Relevo de log|FFT(ρ)| na fatia kz=0 — altura = intensidade", fontsize=13)
plt.tight_layout()
plt.savefig("bravais_outputs_3d/pure_fft_relief.png", dpi=130, bbox_inches="tight")
plt.close()

# teste 2: casca 3D do espectro no volume k inteiro — a forma sólida.
# se for pirâmide/bipirâmide, a casca fecha com faces planas e vértices nos eixos.
log_fft3 = np.log1p(fft3)
try:
    from skimage import measure as _measure

    fig = plt.figure(figsize=(15, 10))
    levels = [np.percentile(log_fft3, p) for p in (50, 75, 90)]
    shell_views = [(25, 45), (25, -45), (90, -90)]
    idx = 1
    for lev in levels:
        vv, ff_, _, _ = _measure.marching_cubes(log_fft3, level=lev)
        vv = (vv - N / 2) * (2 * np.pi / L)
        for elev, azim in shell_views:
            ax = fig.add_subplot(3, 3, idx, projection="3d")
            ax.plot_trisurf(vv[:, 0], vv[:, 1], ff_, vv[:, 2],
                            cmap="viridis", lw=0, alpha=0.8)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title("nível p%d | elev=%d az=%d" % (
                [50, 75, 90][levels.index(lev)], elev, azim), fontsize=9)
            ax.tick_params(labelsize=6)
            idx += 1
    plt.suptitle("Casca 3D de log|FFT(ρ)| no volume k — 3 níveis × 3 ângulos", fontsize=13)
    plt.tight_layout()
    plt.savefig("bravais_outputs_3d/pure_fft_shell.png", dpi=130, bbox_inches="tight")
    plt.close()
except ImportError:
    print("(casca 3D pulada: instale scikit-image para pure_fft_shell.png)")

# teste 3 (numérico, sem olho): a pirâmide é o conjunto |kx|+|ky|+|kz| <= c.
# medimos a intensidade média em cascas de raio L1 (|k|_1) e de raio L2 (|k|_2).
# se a forma for pirâmide, o perfil em L1 é degrau/reta mais nítida que em L2.
k1d = np.fft.fftshift(np.fft.fftfreq(N, d=dx)) * 2 * np.pi
A, B, C = np.meshgrid(k1d, k1d, k1d, indexing="ij")
R_L1 = np.abs(A) + np.abs(B) + np.abs(C)
R_L2 = np.sqrt(A**2 + B**2 + C**2)
nb = 40
prof = {}
for name, R in [("L1 (pirâmide)", R_L1), ("L2 (esfera)", R_L2)]:
    edges = np.linspace(0, R.max(), nb + 1)
    which = np.digitize(R.ravel(), edges) - 1
    sums = np.bincount(which, weights=log_fft3.ravel(), minlength=nb + 1)[:nb]
    cnts = np.bincount(which, minlength=nb + 1)[:nb] + 1e-12
    prof[name] = (0.5 * (edges[:-1] + edges[1:]), sums / cnts)
fig, ax = plt.subplots(figsize=(8, 5))
for name, (rr, pp) in prof.items():
    ax.plot(rr / rr.max(), pp / pp.max(), label=name, lw=2)
ax.set_xlabel("raio normalizado da casca")
ax.set_ylabel("intensidade média normalizada")
ax.legend()
ax.grid(alpha=0.3)
ax.set_title("Perfil do espectro por cascas: L1 vs L2\n(queda mais reta/nítida em L1 → forma piramidal)")
plt.tight_layout()
plt.savefig("bravais_outputs_3d/pure_fft_pyramid_test.png", dpi=140, bbox_inches="tight")
plt.close()

# ------------------------------------------------ 5a-3) mapeamentos extras (mesmos números)

# (1) ANDARES: fatias do espectro em vários kz
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
kz_idx = [N // 2 + off for off in (0, N // 16, N // 8, 3 * N // 16, N // 4, 5 * N // 16, 3 * N // 8, 7 * N // 16)]
vmax = np.log1p(fft3).max()
for ax, iz in zip(axes.flat, kz_idx):
    ax.imshow(np.log1p(fft3[:, :, iz]).T, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
    ax.set_title("kz = %.2f" % ((iz - N // 2) * 2 * np.pi / L), fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
plt.suptitle("Andares do espectro — como o losango muda subindo em kz", fontsize=12)
plt.tight_layout()
plt.savefig("bravais_outputs_3d/pure_fft_floors.png", dpi=130, bbox_inches="tight")
plt.close()

# (2) RAIOS DIRECIONAIS: eixos vs diagonal de face vs diagonal de corpo
c = N // 2
nmax = N // 2 - 1
ray_axis, ray_face, ray_body = [], [], []
for i in range(nmax):
    ax6 = [log_fft3[c + i, c, c], log_fft3[c - i, c, c],
           log_fft3[c, c + i, c], log_fft3[c, c - i, c],
           log_fft3[c, c, c + i], log_fft3[c, c, c - i]]
    fc12 = [log_fft3[c + s1 * i, c + s2 * i, c] for s1 in (1, -1) for s2 in (1, -1)]
    fc12 += [log_fft3[c + s1 * i, c, c + s2 * i] for s1 in (1, -1) for s2 in (1, -1)]
    fc12 += [log_fft3[c, c + s1 * i, c + s2 * i] for s1 in (1, -1) for s2 in (1, -1)]
    bd8 = [log_fft3[c + s1 * i, c + s2 * i, c + s3 * i]
           for s1 in (1, -1) for s2 in (1, -1) for s3 in (1, -1)]
    ray_axis.append(np.mean(ax6))
    ray_face.append(np.mean(fc12))
    ray_body.append(np.mean(bd8))
kstep = 2 * np.pi / L
r_ax = np.arange(nmax) * kstep
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(r_ax, ray_axis, label="ao longo dos eixos (6 raios)", lw=2)
ax.plot(r_ax * np.sqrt(2), ray_face, label="diagonais de face (12 raios)", lw=2)
ax.plot(r_ax * np.sqrt(3), ray_body, label="diagonais de corpo (8 raios)", lw=2)
ax.set_xlabel("|k| real ao longo do raio")
ax.set_ylabel("log|FFT| médio")
ax.legend(); ax.grid(alpha=0.3)
ax.set_title("Intensidade por direção — pontas e braços medidos")
plt.tight_layout()
plt.savefig("bravais_outputs_3d/pure_fft_rays.png", dpi=140, bbox_inches="tight")
plt.close()

# (3) AUTOCORRELAÇÃO: mesmos números de volta no espaço real
acorr = np.fft.fftshift(np.fft.ifftn(np.abs(np.fft.fftn(rho_f)) ** 2).real)
acorr = acorr / acorr.max()
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
axes[0].imshow(acorr[:, :, N // 2].T, origin="lower", cmap="magma",
               extent=[-L / 2, L / 2, -L / 2, L / 2])
axes[0].set_title("autocorrelação — plano xy")
edges_r = np.linspace(0, L / 2, 60)
off = (np.arange(N) - N // 2) * dx
shifted = np.sqrt(off[:, None, None] ** 2 + off[None, :, None] ** 2 + off[None, None, :] ** 2)
which_r = np.digitize(shifted.ravel(), edges_r) - 1
nbin = len(edges_r) - 1
sums_r = np.bincount(which_r, weights=acorr.ravel(), minlength=nbin + 2)[:nbin]
cnts_r = np.bincount(which_r, minlength=nbin + 2)[:nbin] + 1e-12
prof_r = sums_r / cnts_r
rmid = 0.5 * (edges_r[:-1] + edges_r[1:])
axes[1].plot(rmid, prof_r, lw=2)
axes[1].axhline(0, color="gray", lw=0.8)
axes[1].set_xlabel("distância r"); axes[1].set_ylabel("correlação média")
axes[1].set_title("correlação vs distância\n(pico fora de r=0 = distância preferida)")
axes[1].grid(alpha=0.3)
axes[2].plot(rmid, prof_r, lw=2)
axes[2].set_xlim(0, 8); axes[2].set_ylim(prof_r[1:40].min() * 1.5, prof_r[1] * 1.1)
axes[2].axhline(0, color="gray", lw=0.8)
axes[2].set_title("zoom curta distância")
axes[2].grid(alpha=0.3)
plt.suptitle("Autocorrelação da densidade — mesmos números, espaço real", fontsize=12)
plt.tight_layout()
plt.savefig("bravais_outputs_3d/pure_autocorr.png", dpi=140, bbox_inches="tight")
plt.close()

# (4) SIMETRIA MEDIDA: correlação com cópias giradas/espelhadas
def _corr(a, b):
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))

sym_tests = {
    "rot90 xy": _corr(log_fft3, np.rot90(log_fft3, 1, axes=(0, 1))),
    "rot90 xz": _corr(log_fft3, np.rot90(log_fft3, 1, axes=(0, 2))),
    "rot90 yz": _corr(log_fft3, np.rot90(log_fft3, 1, axes=(1, 2))),
    "espelho x": _corr(log_fft3, log_fft3[::-1, :, :]),
    "espelho y": _corr(log_fft3, log_fft3[:, ::-1, :]),
    "espelho z": _corr(log_fft3, log_fft3[:, :, ::-1]),
    "troca x<->y": _corr(log_fft3, np.transpose(log_fft3, (1, 0, 2))),
    "troca x<->z": _corr(log_fft3, np.transpose(log_fft3, (2, 1, 0))),
}
fig, ax = plt.subplots(figsize=(8, 4.5))
names = list(sym_tests.keys())
vals = [sym_tests[k] for k in names]
ax.barh(names, vals, color="#3a506b")
ax.set_xlim(0, 1.0)
ax.set_xlabel("correlação com o original (1 = idêntico)")
ax.set_title("Quanto o espectro é igual a si mesmo girado/espelhado")
ax.grid(alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig("bravais_outputs_3d/pure_symmetry.png", dpi=140, bbox_inches="tight")
plt.close()
print("simetrias:", {k: round(v, 3) for k, v in sym_tests.items()})

# (5) FORMAÇÃO NO TEMPO: perfil L1 do espectro em cada snapshot
fig, ax = plt.subplots(figsize=(8, 5))
edges_l1 = np.linspace(0, R_L1.max(), nb + 1)
which_l1 = np.digitize(R_L1.ravel(), edges_l1) - 1
cnts_l1 = np.bincount(which_l1, minlength=nb + 1)[:nb] + 1e-12
for t in times:
    lf = np.log1p(np.abs(np.fft.fftshift(np.fft.fftn(snapshots[t]))))
    sums_t = np.bincount(which_l1, weights=lf.ravel(), minlength=nb + 1)[:nb]
    pp = sums_t / cnts_l1
    ax.plot(0.5 * (edges_l1[:-1] + edges_l1[1:]) / edges_l1[-1], pp / pp.max(),
            label="t=%d" % t, alpha=0.85)
ax.set_xlabel("raio L1 normalizado")
ax.set_ylabel("intensidade média normalizada")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.set_title("Nascimento da beirada — perfil L1 do espectro por snapshot")
plt.tight_layout()
plt.savefig("bravais_outputs_3d/pure_fft_time.png", dpi=140, bbox_inches="tight")
plt.close()

# projeções de intensidade máxima ao longo de cada eixo (achatamentos laterais)
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, (proj, lab) in zip(
    axes,
    [
        (power3.max(axis=2), "achatado em kz (visto de cima)"),
        (power3.max(axis=1), "achatado em ky (visto de frente)"),
        (power3.max(axis=0), "achatado em kx (visto de lado)"),
    ],
):
    ax.imshow(np.log1p(proj).T, origin="lower", cmap="viridis")
    ax.set_title(lab, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
plt.suptitle("Projeção de máximo do espectro 3D (todo o volume, não só a fatia central)", fontsize=12)
plt.tight_layout()
plt.savefig("bravais_outputs_3d/pure_fft_projections.png", dpi=140, bbox_inches="tight")
plt.close()

# ------------------------------------------------ 5b) FFT geométrico preto e branco
# binário: pixel passa do limiar ou não. Sem colormap, sem interpretação.
sl_kxky = fft3[:, :, N // 2]
power = sl_kxky**2
percs = [99.5, 98.0, 90.0]
fig, axes = plt.subplots(1, len(percs) + 1, figsize=(4.2 * (len(percs) + 1), 4.2))
axes[0].imshow(np.log1p(sl_kxky).T, origin="lower", cmap="gray")
axes[0].set_title("log|FFT| (tons de cinza)")
for ax, p in zip(axes[1:], percs):
    thr_k = np.percentile(power, p)
    ax.imshow((power > thr_k).T, origin="lower", cmap="gray_r")
    ax.set_title("top %.1f%% da potência" % (100 - p))
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
plt.suptitle("Espectro kx-ky binarizado — geometria sem colormap", fontsize=12)
plt.tight_layout()
plt.savefig("bravais_outputs_3d/pure_fft_bw.png", dpi=140, bbox_inches="tight")
plt.close()

# fatia da densidade também em binário
fig, axes = plt.subplots(1, len(percs) + 1, figsize=(4.2 * (len(percs) + 1), 4.2))
sl_rho = rho_f[:, :, N // 2]
axes[0].imshow(sl_rho.T, origin="lower", cmap="gray")
axes[0].set_title(r"$|\Psi|^2$ z=0 (cinza)")
for ax, p in zip(axes[1:], percs):
    thr_r = np.percentile(sl_rho, p)
    ax.imshow((sl_rho > thr_r).T, origin="lower", cmap="gray_r")
    ax.set_title("top %.1f%% da densidade" % (100 - p))
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
plt.suptitle("Densidade z=0 binarizada", fontsize=12)
plt.tight_layout()
plt.savefig("bravais_outputs_3d/pure_density_bw.png", dpi=140, bbox_inches="tight")
plt.close()

# dados brutos para reanálise sem rodar de novo
np.savez_compressed(
    "bravais_outputs_3d/final_state.npz",
    rho_f=rho_f, v=Z["v"], ell=Z["ell"],
    order=np.array(history["order"]), norm=np.array(history["norm"]),
)

# ------------------------------------------------ 6) séries temporais
fig, axes = plt.subplots(3, 2, figsize=(11, 9))
axes[0, 0].plot(history["order"], color="#ee6c4d")
axes[0, 0].set_title("ordem (energia modal relativa)")
axes[0, 1].plot(history["norm"], color="#3a506b")
axes[0, 1].set_title(r"norma $\|\psi\|$ (não fixada)")
axes[1, 0].plot(history["Lambda"], color="#5bc0be", label="Λ")
axes[1, 0].plot(history["alpha"], color="#f4d35e", label="α", alpha=0.8)
axes[1, 0].plot(history["Gamma"], color="#ee6c4d", label="Γ", alpha=0.8)
axes[1, 0].legend(fontsize=8)
axes[1, 0].set_title("coeficientes emergentes (mesma proposta)")
axes[1, 1].plot(history["ell_norm"], color="#1c2541")
axes[1, 1].set_title(r"$|\ell(t)|$ campo latente de vida")
v_arr = np.array(history["v"])
for j in range(8):
    axes[2, 0].plot(v_arr[:, j], alpha=0.85, label="v%d" % j)
axes[2, 0].legend(ncol=4, fontsize=7)
axes[2, 0].set_title("parâmetros de rede v (emergentes, 3D)")
axes[2, 1].plot(history["mean_rho"], label="⟨ρ⟩")
axes[2, 1].plot(history["std_rho"], label="std ρ")
axes[2, 1].legend(fontsize=8)
axes[2, 1].set_title("densidade média / flutuação")
for ax in axes.flat:
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("passo")
plt.suptitle("Dinâmica acoplada 3D — versão restaurada", fontsize=13)
plt.tight_layout()
plt.savefig("bravais_outputs_3d/pure_timeseries.png", dpi=140, bbox_inches="tight")
plt.close()

print("Imagens salvas em bravais_outputs_3d/:")
outs = ["pure_evolution.png", "pure_final_slices.png", "pure_final_peaks3d.png",
        "pure_fft.png", "pure_timeseries.png"]
if has_iso:
    outs.insert(3, "pure_isosurface.png")
for f in outs:
    print("  -", f)

print("\nResumo final:")
print("  order     =", history["order"][-1])
print("  |ell|     =", history["ell_norm"][-1])
print("  Lambda    =", history["Lambda"][-1], "(emergiu, pode ser + ou -)")
print("  ||psi||   =", history["norm"][-1], "(não foi forçada a alvo)")
print("  picos 3D  =", len(peaks))
print("  guarda    =", guard_count["n"], "ativações (0 = proteção ficou muda)")
print("  v final   =", np.round(Z["v"], 3))
