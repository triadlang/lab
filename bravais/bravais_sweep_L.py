#!/usr/bin/env python3
"""
Teste da escala própria (varredura de caixa)
============================================
Hipótese: a beirada c do espectro (orçamento |kx|+|ky|+|kz| <= c) é uma
escala INTERNA do sistema, não herdada do recipiente.

Teste: rodar a MESMA equação em caixas L = 24, 32, 48, mantendo a resolução
dx constante (N cresce junto: 48, 64, 96). Assim a grade de k cobre o mesmo
alcance em todas as rodadas e só o tamanho físico da caixa muda.

  - se c (em unidades físicas de k) ficar no mesmo lugar -> escala própria
  - se c escalar com a caixa -> a beirada era do recipiente

Nada da dinâmica foi alterado: mesmas equações, mesmos coeficientes
emergentes, mesma guarda. Só L e N mudam entre rodadas.

Uso:  python3 bravais_sweep_L.py            (padrão: 800 passos por caixa)
      STEPS=400 python3 bravais_sweep_L.py  (mais rápido, menos maduro)
Aviso: a caixa L=48 usa N=96^3 e é a mais pesada (pode levar bem mais tempo).
"""
import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(None)
os.makedirs("bravais_outputs_3d", exist_ok=True)

STEPS = int(os.environ.get("STEPS", 800))
DT = 0.025
DX = 0.5  # resolução fixa em todas as caixas

CASES = [
    (24.0, 48),
    (32.0, 64),
    (48.0, 96),
]


def run_case(L, N, steps, dt):
    """Mesma dinâmica do bravais_puro_3d.py, parametrizada por (L, N)."""
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

    Z = {
        "h": np.zeros(4), "b": np.zeros(4), "a": np.zeros(4), "n": np.zeros(4),
        "g": np.zeros(4), "ell": np.zeros(8), "v": np.zeros(8),
        "psi": multi_gaussian_field(12),
    }
    welford = {"mean": np.zeros(8), "M2": np.zeros(8), "count": 0}
    guard_n = [0]

    def welford_update(m):
        welford["count"] += 1
        c = welford["count"]
        d = m - welford["mean"]
        welford["mean"] += d / c
        welford["M2"] += d * (m - welford["mean"])

    def welford_z(m):
        c = max(welford["count"], 1)
        var = welford["M2"] / max(c - 1, 1)
        return (m - welford["mean"]) / (np.sqrt(var) + 1e-8)

    def soft_bound(vec, cap=1.0):
        nrm = np.linalg.norm(vec)
        return vec / (1.0 + nrm / max(cap, 1e-12))

    def project_state_vector(vec, radius=2.0):
        nrm = np.linalg.norm(vec)
        return vec if nrm <= radius else vec * (radius / nrm)

    def psi_numeric_guard(psi):
        nrm2 = np.sum(np.abs(psi) ** 2) * dV
        if np.isfinite(nrm2) and nrm2 > 100.0:
            guard_n[0] += 1
        if not np.isfinite(nrm2) or nrm2 > 1e6:
            psi = np.nan_to_num(psi, nan=0.0, posinf=0.0, neginf=0.0)
            nrm2 = np.sum(np.abs(psi) ** 2) * dV
        if nrm2 < 1e-12:
            psi = psi + 1e-3 * (np.random.randn(N, N, N) + 1j * np.random.randn(N, N, N))
            nrm2 = np.sum(np.abs(psi) ** 2) * dV
        if nrm2 > 100.0:
            psi = psi * np.sqrt(50.0 / nrm2)
        return psi

    def experience_signature(psi, Zs):
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
        measures = np.array([
            mean_rho, std_rho, order_energy, rough,
            np.linalg.norm(Zs["ell"]), np.linalg.norm(Zs["a"]),
            np.linalg.norm(Zs["n"]), np.linalg.norm(Zs["v"]),
        ])
        welford_update(measures)
        z = welford_z(measures)
        chi = np.zeros(16)
        chi[:8] = z
        chi[8:12] = Zs["ell"][:4]
        chi[12:16] = 0.5 * (Zs["a"] + Zs["n"])
        chi = chi / (np.sqrt(np.mean(chi**2)) + 1e-8)
        return chi

    def coupled_proposal(chi, Zm):
        ell = Zm["ell"]; a = Zm["a"]; n = Zm["n"]; g = Zm["g"]; v = Zm["v"]
        drive = np.tanh(chi[:8] + 0.2 * ell)
        prop = {
            "h": drive[:4], "b": drive[4:8],
            "a": np.tanh(chi[:4] + 0.3 * a + 0.1 * n),
            "n": np.tanh(chi[4:8] + 0.3 * n + 0.1 * a),
            "g": np.tanh(chi[:4] + 0.25 * g + 0.1 * Zm["b"]),
            "ell": np.tanh(chi[:8] + 0.2 * ell + 0.05 * np.concatenate([a, n])),
            "v": np.tanh(np.concatenate([chi[2:8], chi[0:2]]) + 0.15 * v),
        }
        radial = np.exp(-(X**2 + Y**2 + Zg**2) / (0.25 * L**2 + 1e-8))
        e0, e1, e2, e3 = ell[0], ell[1], ell[2], ell[3]
        v0, v1, v2, v3, v4, v5, v6, v7 = v
        prop["Lambda"] = np.tanh(e0 + 0.3 * chi[0] + 0.2 * a[0])
        prop["alpha"] = 0.5 * (1.0 + np.tanh(e1 + 0.3 * chi[1]))
        prop["Gamma"] = 0.5 * (1.0 + np.tanh(e2 + 0.3 * chi[2]))
        prop["eta"] = 0.5 * (1.0 + np.tanh(e3 + 0.3 * chi[3]))
        prop["sigma"] = 1.0 + 0.5 * np.tanh(v0 + chi[4])
        fx = (2.0 * np.pi / L) * (3.0 + 2.0 * np.tanh(v0))
        fy = (2.0 * np.pi / L) * (3.0 + 2.0 * np.tanh(v1))
        fz = (2.0 * np.pi / L) * (3.0 + 2.0 * np.tanh(v2))
        fxy = (2.0 * np.pi / L) * (2.0 + 2.0 * np.tanh(v3))
        fyz = (2.0 * np.pi / L) * (2.0 + 2.0 * np.tanh(v4))
        fzx = (2.0 * np.pi / L) * (2.0 + 2.0 * np.tanh(v5))
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
        prop["V_mem"] = V_scale * V_mem
        prop["V_ext"] = 0.15 * np.tanh(g[1]) * (
            (X / (L / 2)) ** 2 + (Y / (L / 2)) ** 2 + (Zg / (L / 2)) ** 2
        )
        return prop

    def psi_flow(psi, prop, dt):
        rho = np.abs(psi) ** 2
        V = prop["V_ext"] + prop["Lambda"] * rho + prop["V_mem"]
        psi_k = np.fft.fftn(psi)
        kinetic = np.fft.ifftn(-0.5 * K2 * psi_k)
        frac = prop["alpha"] * np.fft.ifftn((Kabs ** prop["sigma"]) * psi_k)
        damp = -prop["Gamma"] * 0.05 * psi
        noise = prop["eta"] * 0.02 * (
            np.random.randn(N, N, N) + 1j * np.random.randn(N, N, N)
        )
        return psi + dt * (-1j * (kinetic + V * psi + frac) + damp + noise)

    def coupled_step(Z, dt, max_iter=5):
        Z0 = {k: (val.copy() if isinstance(val, np.ndarray) else val) for k, val in Z.items()}
        Zc = {k: (val.copy() if isinstance(val, np.ndarray) else val) for k, val in Z.items()}
        for _ in range(max_iter):
            Zm = {k: 0.5 * (Z0[k] + Zc[k]) for k in Z0}
            chi = experience_signature(Zm["psi"], Zm)
            prop = coupled_proposal(chi, Zm)
            keys = ["h", "b", "a", "n", "g", "ell", "v"]
            raw = {k: prop[k] - Zm[k] for k in keys}
            stack = soft_bound(np.concatenate([raw[k] for k in keys]), cap=1.0)
            i0 = 0
            for k in keys:
                nsz = raw[k].size
                Zc[k] = project_state_vector(Z0[k] + stack[i0:i0 + nsz], radius=2.5)
                i0 += nsz
            Zc["psi"] = psi_numeric_guard(psi_flow(Zm["psi"], prop, dt))
            if np.linalg.norm(Zc["ell"] - Zm["ell"]) < 1e-3:
                break
        return Zc

    for t in range(steps):
        Z = coupled_step(Z, dt=dt)
        if t % 100 == 0:
            nrm = np.sqrt(np.sum(np.abs(Z["psi"]) ** 2) * dV)
            print("  L=%.0f N=%d | t=%4d | ‖ψ‖=%.3f" % (L, N, t, nrm))
    rho_f = np.abs(Z["psi"]) ** 2
    return rho_f, guard_n[0]


def edge_profile_L1(rho_f, L, N, nb=60):
    """Perfil de cascas L1 do espectro, em unidades FÍSICAS de k."""
    dx = L / N
    lf = np.log1p(np.abs(np.fft.fftshift(np.fft.fftn(rho_f))))
    k1d = np.fft.fftshift(np.fft.fftfreq(N, d=dx)) * 2 * np.pi
    A, B, C = np.meshgrid(k1d, k1d, k1d, indexing="ij")
    R_L1 = np.abs(A) + np.abs(B) + np.abs(C)
    edges = np.linspace(0, R_L1.max(), nb + 1)
    which = np.digitize(R_L1.ravel(), edges) - 1
    sums = np.bincount(which, weights=lf.ravel(), minlength=nb + 1)[:nb]
    cnts = np.bincount(which, minlength=nb + 1)[:nb] + 1e-12
    prof = sums / cnts
    rmid = 0.5 * (edges[:-1] + edges[1:])
    return rmid, prof


def edge_position(rmid, prof):
    """Posição c da beirada: ponto de descida mais íngreme do perfil
    (ignorando o primeiro bin, que contém o pico k=0)."""
    p = prof / (prof.max() + 1e-30)
    grad = np.gradient(p, rmid)
    i0 = 2  # ignora a agulha central
    i_edge = i0 + int(np.argmin(grad[i0:]))
    return rmid[i_edge]


results = {}
print("Varredura de caixa — mesma equação, dx fixo = %.2f, %d passos por caixa" % (DX, STEPS))
for L, N in CASES:
    assert abs(L / N - DX) < 1e-9, "dx deve ser igual em todas as caixas"
    print("caixa L=%.0f (N=%d^3) ..." % (L, N))
    rho_f, gn = run_case(L, N, STEPS, DT)
    rmid, prof = edge_profile_L1(rho_f, L, N)
    c = edge_position(rmid, prof)
    results[L] = {"rmid": rmid, "prof": prof, "c": c, "guard": gn, "N": N}
    print("  -> beirada c = %.3f (k físico) | guarda ativou %d vezes" % (c, gn))
    np.savez_compressed(
        "bravais_outputs_3d/sweep_L%.0f.npz" % L,
        rho_f=rho_f, rmid=rmid, prof=prof, c=c,
    )

# ---------------------------------------------------------------- veredito
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for L in sorted(results):
    r = results[L]
    axes[0].plot(r["rmid"], r["prof"] / r["prof"].max(),
                 lw=2, label="L=%.0f (c=%.2f)" % (L, r["c"]))
    axes[0].axvline(r["c"], ls="--", alpha=0.4)
axes[0].set_xlabel("raio L1 em k físico (|kx|+|ky|+|kz|)")
axes[0].set_ylabel("intensidade média normalizada")
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[0].set_title("Perfis L1 — eixo em unidades físicas, caixas diferentes")

Ls = sorted(results)
cs = [results[L]["c"] for L in Ls]
axes[1].plot(Ls, cs, "o-", lw=2, ms=10, color="#ee6c4d", label="c medido")
c_ref = cs[Ls.index(32.0)] if 32.0 in Ls else cs[0]
axes[1].plot(Ls, [c_ref * (32.0 / L) for L in Ls], "s--", alpha=0.6,
             color="#3a506b", label="se escalasse com a caixa (~1/L)")
axes[1].plot(Ls, [c_ref] * len(Ls), "^--", alpha=0.6,
             color="#5bc0be", label="se for escala própria (constante)")
axes[1].set_xlabel("tamanho da caixa L")
axes[1].set_ylabel("posição da beirada c")
axes[1].legend(); axes[1].grid(alpha=0.3)
axes[1].set_title("Veredito: c acompanha a caixa ou é do sistema?")
plt.suptitle("Teste da escala própria — a beirada pertence ao sistema ou ao recipiente?", fontsize=13)
plt.tight_layout()
plt.savefig("bravais_outputs_3d/sweep_verdict.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nResumo:")
for L in Ls:
    print("  L=%4.0f | c = %.3f | guarda = %d" % (L, results[L]["c"], results[L]["guard"]))
spread = (max(cs) - min(cs)) / (np.mean(cs) + 1e-30)
print("variação relativa de c entre caixas: %.1f%%" % (100 * spread))
print("  (pequena -> escala própria; comparável a ~1/L -> herdada do recipiente)")
print("figura: bravais_outputs_3d/sweep_verdict.png")
