import numpy as np
import json

# Parâmetro de frequência da EDP (Eq. 6.3)
a = 1


def generate_analytical_solution(a=a, num_points=1024, filename=None):
    """
    Gera e salva os pontos de avaliação e a solução analítica exata
    para a equação de Poisson 1D de alta frequência (Eq. 6.3) em formato JSON:
        u(x) = sin(2 * pi * x) + 0.1 * sin(a * pi * x),  x in [0, 1]
    """
    if filename is None:
        filename = f"gt_poisson_1d_high_freq_a{a}.json"

    print(f"Gerando malha de teste com {num_points} pontos para a = {a}...")

    # Cria o array de coordenadas (coluna)
    x_eval = np.linspace(0.0, 1.0, num_points).reshape(-1, 1)

    # Calcula a solução exata (Eq. 6.3)
    # u(x) = sin(2 * pi * x) + 0.1 * sin(a * pi * x)
    u_eval = np.sin(2.0 * np.pi * x_eval) + 0.1 * np.sin(a * np.pi * x_eval)

    # Estrutura os dados no formato JSON
    data = {
        "metadata": {
            "num_points": num_points,
            "domain": "x in [0, 1]",
            "a": a
        },
        "X": x_eval.tolist(),
        "U_true": u_eval.tolist()
    }

    # Salva os dados no arquivo JSON
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"Arquivo '{filename}' gerado com sucesso!")


if __name__ == "__main__":
    generate_analytical_solution(a=a)
