import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

gt_file = "gt_poisson_1d.json"
arch_str = sys.argv[1] if len(sys.argv) > 1 else "1_20_20_20_1"

methods = {
    "PINN": ("pinn_poisson_1d", "pinn_1d", "crimson"),
    "VPINN-R1": ("vpinn_poisson_1d", "vpinn_poisson_1d_R1", "darkorange"),
    "VPINN-R2": ("vpinn_poisson_1d", "vpinn_poisson_1d_R2", "seagreen"),
    "SV-PINN": ("svpinn_poisson_1d", "svpinn_1d", "royalblue"),
}

with open(gt_file, "r") as f:
    gt = json.load(f)
x_gt = np.array(gt["X"]).flatten()
u_true = np.array(gt["U_true"]).flatten()

fig, ax = plt.subplots(figsize=(8, 5))

for label, (dir_name, file_prefix, color) in methods.items():
    results_dir = Path(dir_name)
    pontos_file = results_dir / f"pontos_{file_prefix}_{arch_str}.json"
    if not pontos_file.exists():
        print(f"[aviso] {pontos_file} não encontrado, pulando {label}")
        continue

    with open(pontos_file, "r") as f:
        pontos = json.load(f)
    x_nn = np.array(pontos["x"]).flatten()
    u_nn = np.array(pontos["y_nn"]).flatten()

    if x_nn.shape[0] != x_gt.shape[0] or not np.allclose(x_gt, x_nn):
        order = np.argsort(x_nn)
        u_nn = np.interp(x_gt, x_nn[order], u_nn[order])

    erro = np.abs(u_true - u_nn)
    ax.plot(x_gt, erro, color=color, lw=1.5, label=label)

    print(f"{label}: L2 relativo = {np.linalg.norm(erro) / np.linalg.norm(u_true):.6e} | "
          f"erro max = {erro.max():.6e}")

ax.set_xlabel("x")
ax.set_ylabel(r"$|u_{true}(x) - u_{modelo}(x)|$")
ax.set_title(f"Erro pontual — arquitetura {arch_str}")
ax.set_yscale("log")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()

out_path = Path(f"erro_pontual_comparativo_1d_{arch_str}.png")
fig.savefig(out_path, dpi=200)
print(f"Figura salva em {out_path}")