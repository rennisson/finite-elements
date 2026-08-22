# -*- coding: utf-8 -*-
"""
plot_1d_poisson_vpinn_erro_pontual.py

Plota o erro ponto-a-ponto |u_NN(x) - u_exact(x)| em escala log, separando os
gráficos pelo método ("vpinnR1", "vpinnR2") para o problema de Poisson 1D no intervalo (0, 1).
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "."

# Atualizado para a tag gerada pelo script de intervalo (0,1)
EXACT_SOLUTIONS = ["unit_interval"]
METHODS = ["vpinnR1", "vpinnR2"]
HIDDEN_LAYER_CONFIGS = [1, 2]

# Solução exata para o problema no intervalo (0,1)
SOLUTION_TITLES = {
    "unit_interval": r"$u^{exact}(x) = x \exp(-x^2)$",
}

# Mesmo esquema de cores/estilo usado em plot_1d_poisson_vpinn.py
STYLE = {
    ("vpinnR1", 1): dict(color="#1f77b4", linestyle="-", label=r"VPINN $\mathcal{R}^{(1)}$, L=1"),
    ("vpinnR1", 2): dict(color="#1f77b4", linestyle="--", label=r"VPINN $\mathcal{R}^{(1)}$, L=2"),
    ("vpinnR2", 1): dict(color="#ff7f0e", linestyle="-", label=r"VPINN $\mathcal{R}^{(2)}$, L=1"),
    ("vpinnR2", 2): dict(color="#ff7f0e", linestyle="--", label=r"VPINN $\mathcal{R}^{(2)}$, L=2"),
}

EPS = 1e-16  # evita log(0) no ponto-a-ponto quando o erro e numericamente nulo


def plot_pointwise_error():
    # Loop 1: Itera sobre as soluções exatas
    for solution_name in EXACT_SOLUTIONS:
        
        # Loop 2: Itera sobre os métodos (R1 e R2), criando uma figura para CADA método
        for method in METHODS:
            fig, ax = plt.subplots(figsize=(8, 6))
            any_data_plotted = False

            # Loop 3: Plota as configurações de camadas ocultas (L=1, L=2) na mesma figura
            for L in HIDDEN_LAYER_CONFIGS:
                tag = f"{solution_name}_{method}_L{L}"
                nome_arquivo = os.path.join(RESULTS_DIR, f"pontos_vpinn_1d_{tag}.json")

                if not os.path.exists(nome_arquivo):
                    print(f"Arquivo nao encontrado: {nome_arquivo}")
                    continue

                with open(nome_arquivo, "r") as f:
                    data = json.load(f)

                x = np.array(data["x"])
                y_nn = np.array(data["y_nn"])
                y_exact = np.array(data["y_exact"])

                point_wise_error = np.abs(y_nn - y_exact)
                point_wise_error = np.maximum(point_wise_error, EPS)

                style = STYLE[(method, L)]
                ax.plot(x, point_wise_error, linewidth=1.3, **style)
                any_data_plotted = True

            # Se nenhum dado foi plotado para essa combinação, fecha a figura e pula
            if not any_data_plotted:
                print(f"Nenhum dado encontrado para '{solution_name}' com '{method}'. Pulando figura.")
                plt.close(fig)
                continue

            # Formatação do gráfico
            ax.set_yscale("log")
            ax.set_xlabel("x")
            ax.set_ylabel("point-wise error  |u_NN(x) - u_exact(x)|")
            
            # Atualização do título para refletir o método atual e o intervalo correto
            nome_metodo_formatado = "R(1)" if method == "vpinnR1" else "R(2)"
            ax.set_title(f"Erro ponto-a-ponto: Poisson 1D (0, 1) - Método {nome_metodo_formatado}\n{SOLUTION_TITLES[solution_name]}",
                         fontsize=11)
            ax.grid(True, which="both", ls="--", alpha=0.4)

            ax.legend(loc="best", borderaxespad=0.)
            plt.tight_layout()

            # Salva o gráfico combinando o nome da solução e o método
            output_img = f"POISSON-1D-VPINN-ERRO-PONTUAL-{solution_name.upper()}-{method.upper()}.png"
            plt.savefig(output_img, dpi=300, bbox_inches="tight")
            print(f"Grafico gerado e salvo como: {output_img}")
            
            # Fecha a figura após salvar para evitar sobreposição na próxima iteração
            plt.close(fig)

if __name__ == "__main__":
    plot_pointwise_error()