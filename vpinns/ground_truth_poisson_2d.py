import numpy as np
import json

def generate_analytical_solution_2d(num_points=1000, filename="gt_poisson_2d.json"):
    """
    Gera e salva os pontos de avaliação e a solução analítica exata 
    para a equação de Poisson 2D do artigo.
    """
    print(f"Gerando malha de teste com {num_points}x{num_points} pontos...")
    
    # Cria os vetores no domínio [0, 1]
    x_eval = np.linspace(0.0, 1.0, num_points)
    y_eval = np.linspace(0.0, 1.0, num_points)
    
    # Cria a malha 2D para as coordenadas (X e Y)
    X, Y = np.meshgrid(x_eval, y_eval)
    
    # Calcula a solução exata do artigo:
    # u_true(x, y) = x^2 * (x - 1)^2 * y * (y - 1)^2
    U_true = (X**2) * ((X - 1)**2) * Y * ((Y - 1)**2)
    
    # Estrutura os dados em um dicionário compatível com JSON
    # As matrizes do numpy precisam ser convertidas para listas com .tolist()
    data = {
        "metadata": {
            "num_points": num_points,
            "domain": "x in [0, 1], y in [0, 1]"
        },
        "X": X.tolist(),
        "Y": Y.tolist(),
        "U_true": U_true.tolist()
    }
    
    # Salva os dados em formato JSON
    with open(filename, 'w') as json_file:
        json.dump(data, json_file)
        
    print(f"Arquivo '{filename}' gerado com sucesso!")

if __name__ == "__main__":
    generate_analytical_solution_2d(num_points=1000)