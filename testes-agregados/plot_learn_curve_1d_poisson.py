# -*- coding: utf-8 -*-
"""
Script para plotar as curvas de aprendizado das PINNs, SV-PINNs e V-PINNs em 1D (Poisson).
Lê os arquivos JSON gerados pelos respectivos scripts de treinamento.
"""
import json
import math
import os
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_all_training_curves(svpinn_dir, pinn_dir, vpinn_dir, output_path):
    # Arquiteturas originais testadas
    architectures = [
        [5, 1], [10, 1], [20, 1], [40, 1],
        [5, 5, 1], [10, 10, 1], [20, 20, 1], [40, 40, 1],
        [5, 5, 5, 1], [10, 10, 10, 1], [20, 20, 20, 1], [40, 40, 40, 1]
    ]
    
    # Configurações de estilo para cada modelo baseadas na sua solicitação
    models_config = {
        "PINN": {
            "dir": pinn_dir,
            "color": "blue",
            "ls": "-",
            "label": "PINN (L-BFGS, 20k passos)",
            "file_pattern": "curva_treino_pinn_1d_{}.json"
        },
        "SVPINN": {
            "dir": svpinn_dir,
            "color": "green",
            "ls": "-",
            "label": "SVPINN (L-BFGS, 5k passos)",
            "file_pattern": "curva_treino_svpinn_1d_{}.json"
        },
        "VPINN_R1": {
            "dir": vpinn_dir,
            "color": "orange",
            "ls": "--",
            "label": "VPINN R1 (L-BFGS, 20k passos)",
            "file_pattern": "curva_treino_vpinn_poisson_1d_R1_{}.json" 
        },
        "VPINN_R2": {
            "dir": vpinn_dir,
            "color": "purple",
            "ls": "--",
            "label": "VPINN R2 (L-BFGS, 20k passos)",
            "file_pattern": "curva_treino_vpinn_poisson_1d_R2_{}.json" 
        }
    }
    
    n = len(architectures)
    ncols = 4
    nrows = math.ceil(n / ncols)
    
    # Prepara a figura com base no número de arquiteturas
    fig, axes = plt.subplots(
        nrows, ncols, 
        figsize=(5 * ncols, 3.8 * nrows), 
        squeeze=False, 
        sharey=True
    )
    
    # Dicionário para rastrear as legendas globais sem repeti-las
    legend_dict = {}

    for j, arch in enumerate(architectures):
        row = j // ncols
        col = j % ncols
        ax = axes[row, col]
        
        # Reconstrói o arch_str (ex: 1_5_1, 1_10_10_1)
        width = [1] + arch
        arch_str = "_".join(map(str, width))
        
        ax.set_title(f"Arquitetura {width}", fontsize=11)
        ax.set_yscale("log")
        
        # Iterar sobre os 3 modelos para plotá-os no mesmo subplot
        for model_name, config in models_config.items():
            json_filename = config["file_pattern"].format(arch_str)
            json_path = os.path.join(config["dir"], json_filename)
            
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
            except FileNotFoundError:
                # Se não achar o arquivo deste modelo em específico, pula para o próximo
                continue
                
            steps = np.asarray(data["steps"])
            err_per_run = np.asarray(data["l2_relative_error_per_run"])
            err_mean = np.asarray(data["l2_relative_error_mean"])
            err_std = np.asarray(data["l2_relative_error_std"])
            num_runs = data.get("num_runs", err_per_run.shape[0])

            # Se houver trajetória do Adam (ex: PINN), prefixa a curva com
            # os passos do Adam, deslocando os passos do L-BFGS para que a
            # curva fique contínua (Adam seguido de L-BFGS).
            if "steps_adam" in data:
                epochs_adam = data.get("epochs_adam", 0)
                steps_adam = np.asarray(data["steps_adam"])
                err_per_run_adam = np.asarray(data["l2_relative_error_per_run_adam"])
                err_mean_adam = np.asarray(data["l2_relative_error_mean_adam"])

                steps = np.concatenate([steps_adam, epochs_adam + steps])
                err_per_run = np.concatenate([err_per_run_adam, err_per_run], axis=1)
                err_mean = np.concatenate([err_mean_adam, err_mean])
            
            # Plotar runs individuais com transparência
            for run in range(num_runs):
                ax.plot(steps, err_per_run[run], color=config["color"], alpha=0.15, 
                        lw=0.8, ls=config["ls"])
                
            # Plotar média e sombra do desvio padrão
            mean_line, = ax.plot(steps, err_mean, color=config["color"], 
                                 lw=2.0, ls=config["ls"])
            # ax.fill_between(steps, err_mean - err_std, err_mean + err_std,
            #                  color=config["color"], alpha=0.1, linewidth=0)
            
            # Guardar a linha principal para a legenda global
            if model_name not in legend_dict:
                legend_dict[model_name] = (mean_line, config["label"])
        
        # Ajustes de visualização do gráfico individual
        if row == nrows - 1 or j + ncols >= n:
            ax.set_xlabel("Passos de Treinamento (Steps)")
        if col == 0:
            ax.set_ylabel("Erro Relativo $L^2$")

    # Remove os subplots vazios na última linha, se a divisão por colunas não for exata
    for j in range(n, nrows * ncols):
        row = j // ncols
        col = j % ncols
        fig.delaxes(axes[row, col])

    # Adiciona a legenda única na parte inferior (abaixo de todos os subplots)
    if legend_dict:
        handles = [v[0] for v in legend_dict.values()]
        labels = [v[1] for v in legend_dict.values()]
        
        fig.legend(
            handles, 
            labels, 
            loc="lower center", 
            bbox_to_anchor=(0.5, -0.05), # Posicionado para baixo da figura
            ncol=len(legend_dict), 
            frameon=True,
            fontsize=12
        )
    
    fig.suptitle("Curvas de Aprendizado: PINNs, SVPINNs e VPINNs (Poisson 1D)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigura unificada salva com sucesso em: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera plots das curvas de treinamento de PINNs, SVPINNs e VPINNs.")
    # Adicionando argumentos de diretório para os três modelos
    parser.add_argument("--svpinn-dir", type=str, default="svpinn_poisson_1d",
                        help="Diretório contendo os arquivos JSON da SVPINN.")
    parser.add_argument("--pinn-dir", type=str, default="pinn_poisson_1d",
                        help="Diretório contendo os arquivos JSON da PINN.")
    parser.add_argument("--vpinn-dir", type=str, default="vpinn_poisson_1d",
                        help="Diretório contendo os arquivos JSON da VPINN.")
    parser.add_argument("--output-file", type=str, default="curvas_treinamento_1d_poisson_todos_modelos.png",
                        help="Nome do arquivo de saída da imagem final.")
    args = parser.parse_args()
    
    plot_all_training_curves(args.svpinn_dir, args.pinn_dir, args.vpinn_dir, args.output_file)