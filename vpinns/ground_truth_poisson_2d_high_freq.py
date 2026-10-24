import numpy as np
import json

# Parâmetro de frequência da EDP (Eq. 6.4): a in {1, 10, 25, 50, 75}
a = 1


def generate_analytical_solution_2d(a=a, num_points=1024, filename=None):
    """
    Gera e salva os pontos de avaliação e a solução analítica exata 
    para a equação de Poisson 2D de alta frequência (Eq. 6.4) em formato JSON:
        u(x, y) = sin(pi * x) * sin(pi * y) * [sin(a * (x + y)) + sin(2 * pi * x) + cos(3 * pi * y)]
    """
    if filename is None:
        filename = f"gt_poisson_2d_high_freq_a{a}.json"

    print(f"Gerando malha de teste com {num_points}x{num_points} pontos para a = {a}...")

    # Cria os vetores no domínio [0, 1]
    x_eval = np.linspace(0.0, 1.0, num_points)
    y_eval = np.linspace(0.0, 1.0, num_points)

    # Cria a malha 2D para as coordenadas (X e Y)
    X, Y = np.meshgrid(x_eval, y_eval)

    # Calcula a solução exata (Eq. 6.4):
    # u_true(x, y) = sin(pi * x) * sin(pi * y) * [sin(a * (x + y)) + sin(2 * pi * x) + cos(3 * pi * y)]
    U_true = (
        np.sin(np.pi * X)
        * np.sin(np.pi * Y)
        * (np.sin(a * (X + Y)) + np.sin(2.0 * np.pi * X) + np.cos(3.0 * np.pi * Y))
    )

    # Estrutura os dados em um dicionário compatível com JSON
    # As matrizes do numpy precisam ser convertidas para listas com .tolist()
    data = {
        "metadata": {
            "num_points": num_points,
            "domain": "x in [0, 1], y in [0, 1]",
            "a": a
        },
        "X": X.tolist(),
        "Y": Y.tolist(),
        "U_true": U_true.tolist()
    }

    # Salva os dados em formato JSON
    with open(filename, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file)

    print(f"Arquivo '{filename}' gerado com sucesso!")


if __name__ == "__main__":
    generate_analytical_solution_2d(a=a, num_points=1024)
