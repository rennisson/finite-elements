import numpy as np
import json

def generate_analytical_solution_3d(num_points=150, filename="gt_poisson_3d_150.json"):
    """
    Gera e salva os pontos de avaliação e a solução analítica exata 
    para a equação de Poisson 3D no formato JSON.
    """
    print(f"Gerando malha de teste com {num_points}x{num_points}x{num_points} pontos...")
    
    # Cria os vetores no domínio [0, 1]
    x_eval = np.linspace(0.0, 1.0, num_points)
    y_eval = np.linspace(0.0, 1.0, num_points)
    z_eval = np.linspace(0.0, 1.0, num_points)
    
    # Cria a malha 3D
    # indexing='ij' mantém coerência com a ordem (x, y, z)
    X, Y, Z = np.meshgrid(x_eval, y_eval, z_eval, indexing='ij')
    
    # Calcula a solução exata conforme o artigo
    # u_true(x, y, z) = sin(pi*x) * sin(pi*y) * sin(pi*z)
    U_true = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
    
    # Estrutura os dados em um dicionário
    data = {
        "metadata": {
            "num_points": num_points,
            "domain": "x in [0, 1], y in [0, 1], z in [0, 1]"
        },
        "X": X.tolist(),
        "Y": Y.tolist(),
        "Z": Z.tolist(),
        "U_true": U_true.tolist()
    }
    
    print("Salvando arquivo JSON (isso pode levar alguns instantes devido ao tamanho)...")
    # Salva os dados em formato JSON
    with open(filename, 'w') as f:
        json.dump(data, f)
        
    print(f"Arquivo '{filename}' gerado com sucesso!")

if __name__ == "__main__":
    generate_analytical_solution_3d(num_points=150)