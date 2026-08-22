import numpy as np
import json

def generate_analytical_solution(num_points=512, filename="gt_poisson_1d.json"):
    """
    Gera e salva os pontos de avaliação e a solução analítica exata 
    para a equação de Poisson 1D em formato JSON.
    """
    print(f"Gerando malha de teste com {num_points} pontos...")
    
    # Cria o array de coordenadas (coluna)
    x_eval = np.linspace(0.0, 1.0, num_points).reshape(-1, 1)
    
    # Calcula a solução exata
    # u(x) = x * exp(-x^2)
    u_eval = x_eval * np.exp(-x_eval**2)
    
    # Estrutura os dados no formato JSON solicitado (adaptado para 1D)
    data = {
        "metadata": {
            "num_points": num_points,
            "domain": "x in [0, 1]"
        },
        "X": x_eval.tolist(),
        "U_true": u_eval.tolist()
    }
    
    # Salva os dados no arquivo JSON
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        
    print(f"Arquivo '{filename}' gerado com sucesso!")

if __name__ == "__main__":
    generate_analytical_solution()