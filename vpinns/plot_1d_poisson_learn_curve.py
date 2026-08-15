# -*- coding: utf-8 -*-
"""
plot_learn_curve_1d_poisson_vpinn.py

Plota a evolucao do erro L2 relativo ao longo do treinamento (Adam) das
VPINNs para o problema de Poisson 1D no intervalo (0, 1), 
separando os graficos por formulacao (R(1) e R(2)).

Curvas L1 sao plotadas em preto e curvas L2 em rosa.
"""
import json
import os
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Atualizado para a tag gerada pelo script de intervalo (0,1)
EXACT_SOLUTIONS = ["unit_interval"]
METHODS = ["vpinnR1", "vpinnR2"]
HIDDEN_LAYER_CONFIGS = [1, 2]

# Dicionario atualizado para definir estilo baseado no numero de camadas (L)
LAYER_STYLE = {
    1: dict(color="black", label="L=1 (1 camada)"),
    2: dict(color="pink", label="L=2 (2 camadas)"),
}

# Atualizado para refletir o nome da solução no problema atual
SOLUTION_TITLES = {
    "unit_interval": r"solução $u(x) = x \exp(-x^2)$",
}

def plot_training_curves(results_dir, solution_name, method, output_path, threshold=0.01):
    fig, ax = plt.subplots(figsize=(7, 5))
    
    legend_handles = []
    legend_labels = []
    any_data_plotted = False

    # Itera sobre as configuracoes de camada (L1 e L2) para o mesmo grafico
    for L in HIDDEN_LAYER_CONFIGS:
        tag = f"{solution_name}_{method}_L{L}"
        json_path = os.path.join(results_dir, f"curva_treino_vpinn_1d_{tag}.json")

        if not os.path.exists(json_path):
            print(f"Arquivo nao encontrado: {json_path}")
            continue

        with open(json_path, "r") as f:
            data = json.load(f)

        steps = np.asarray(data["steps"])
        err_per_run = np.asarray(data["l2_relative_error_per_run"])
        err_mean = np.asarray(data["l2_relative_error_mean"])
        err_std = np.asarray(data["l2_relative_error_std"])
        num_runs = data.get("num_runs", err_per_run.shape[0])

        style = LAYER_STYLE[L]
        color = style["color"]

        # Plota a media e o desvio padrao
        mean_line, = ax.plot(steps, err_mean, color=color, lw=1.8, label=style["label"])
        # ax.fill_between(steps, err_mean - err_std, err_mean + err_std,
        #                  color=color, alpha=0.15, linewidth=0)

        any_data_plotted = True

        if style["label"] not in legend_labels:
            legend_handles.append(mean_line)
            legend_labels.append(style["label"])

    # Finalizacao e formatacao caso algum dado tenha sido plotado
    if any_data_plotted:
        threshold_line = ax.axhline(threshold, color="gray", ls="--", lw=0.8)
        if "Limiar" not in legend_labels:
            legend_handles.append(threshold_line)
            legend_labels.append(f"Limiar ({threshold:.0%})")

        ax.set_yscale("log")
        
        # Formata o titulo para refletir R(1) ou R(2) e o nome correto
        nome_metodo = "R(1)" if method == "vpinnR1" else "R(2)"
        ax.set_title(f"VPINN {nome_metodo} - Poisson 1D ({SOLUTION_TITLES[solution_name]})", fontsize=11)
        ax.set_xlabel("Steps (Adam)")
        ax.set_ylabel(r"$L^2$ relative error")
        ax.grid(True, which="both", ls="--", alpha=0.3)

        if legend_handles:
            ax.legend(legend_handles, legend_labels, loc="best")

        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Figura salva com sucesso em: {output_path}")
    else:
        print(f"Nenhum dado encontrado para '{solution_name}' e '{method}'. Pulando figura.")

    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera plots das curvas de treinamento da VPINN.")
    parser.add_argument("--results-dir", type=str, default=".",
                        help="Diretorio contendo os arquivos JSON (padrao: diretorio atual).")
    args = parser.parse_args()

    # Loops aninhados para separar os arquivos por solucao exata e por metodo (R1 e R2)
    for solution_name in EXACT_SOLUTIONS:
        for method in METHODS:
            output_file = f"curvas_treinamento_1d_poisson_vpinn_{solution_name.upper()}_{method.upper()}.png"
            plot_training_curves(args.results_dir, solution_name, method, output_file)