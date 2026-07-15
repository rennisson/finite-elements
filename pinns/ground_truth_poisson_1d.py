import numpy as np

def generate_analytical_solution(num_points=512, filename="gt_poisson_1d_512.npz"):
    """
    Gera e salva os pontos de avaliação e a solução analítica exata 
    para a equação de Poisson 1D.
    """
    print(f"Gerando malha de teste com {num_points} pontos...")
    
    # Cria o array de coordenadas (coluna)
    x_eval = np.linspace(0.0, 1.0, num_points).reshape(-1, 1)
    
    # Calcula a solução exata
    # u(x) = x * exp(-x^2)
    u_eval = x_eval * np.exp(-x_eval**2)
    
    # Salva os dados em formato compactado do NumPy
    np.savez_compressed(
        filename,
        x_eval=x_eval,
        u_eval=u_eval
    )
    
    print(f"Arquivo '{filename}' gerado com sucesso!")

if __name__ == "__main__":
    generate_analytical_solution()