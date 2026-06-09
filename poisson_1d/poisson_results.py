import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional

def solucao_exata(x: np.ndarray) -> np.ndarray:
    """Calcula a solução analítica exata da equação de Poisson 1D."""
    return x * np.exp(-x**2)

def carregar_dados_pinn(caminho_arquivo: str) -> np.ndarray:
    """Carrega as predições da rede neural (PINN)."""
    with np.load(caminho_arquivo) as dados:
        return dados['y_nn']

def carregar_dados_fem(caminho_arquivo: str) -> tuple[np.ndarray, np.ndarray]:
    """Carrega as coordenadas X e os valores Y do FEniCSx."""
    with np.load(caminho_arquivo) as dados:
        return dados['x_fem'], dados['y_fem']

def plotar_comparativo(
    arquivos_pinn: Dict[str, str], 
    arquivos_fem: Dict[str, str],
    x_teste: np.ndarray, 
    caminho_salvar: Optional[str] = None
):
    """
    Plota as soluções Exata, PINN e FEM no mesmo gráfico.
    """
    plt.figure(figsize=(12, 7))
    
    # 1. Solução Exata
    y_exato = solucao_exata(x_teste)
    plt.plot(
        x_teste, y_exato,
        linestyle='-', color='r', linewidth=5, 
        label='Solução Exata', zorder=1
    )

    cores = ['blue', 'orange', 'green', 'purple', 'cyan', 'brown', 'pink', 'gray', 'olive']
    
    # --- PLOT DO FEM ---
    for idx, (rotulo, caminho) in enumerate(arquivos_fem.items()):
        try:
            x_fem, y_fem = carregar_dados_fem(caminho)
            cor = cores[idx % len(cores)]

            plt.plot(
                x_fem, y_fem,
                linestyle='--', color=cor, linewidth=2,
                label=f"{rotulo}", zorder=3
            )
        except Exception as e:
            print(f"Erro FEM '{rotulo}': {e}")

    # --- PLOT DAS PINNS ---
    for idx, (rotulo, caminho) in enumerate(arquivos_pinn.items()):
        try:
            y_pred = carregar_dados_pinn(caminho)
            cor = cores[idx % len(cores)]
            
            plt.plot(
                x_teste, y_pred,
                linestyle='-', color=cor, linewidth=2,
                label=f"{rotulo}", alpha=0.8, zorder=3
            )
        except Exception as e:
            print(f"Erro PINN '{rotulo}': {e}")
            
    # Formatação Final
    plt.title('PINNs vs FEM', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('u(x)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Posiciona a legenda fora do eixo (grid), à direita e alinhada pelo topo
    plt.legend(
        bbox_to_anchor=(1, 1), 
        loc='upper left', 
        frameon=True, shadow=True, 
        ncol=1
    )
    plt.tight_layout()
    
    if caminho_salvar:
        plt.savefig(caminho_salvar, dpi=300)
        print(f"Gráfico salvo com sucesso em: {caminho_salvar}")
    else:
        plt.show()

def main():
    """
    Função principal que orquestra os dados e chama o plot.
    """
    # Recria o domínio de teste (idêntico ao usado no treinamento original)
    x_test = np.linspace(0, 1, 1000).reshape((1000, 1))
    
    arquivos_fem = {
        'FEM Mesh 64':   'npz/dados_fem_nx64.npz',
        'FEM Mesh 128':  'npz/dados_fem_nx128.npz',
        'FEM Mesh 256':  'npz/dados_fem_nx256.npz',
        'FEM Mesh 512':  'npz/dados_fem_nx512.npz',
        'FEM Mesh 1024': 'npz/dados_fem_nx1024.npz',
        'FEM Mesh 2048': 'npz/dados_fem_nx2048.npz',
        'FEM Mesh 4096': 'npz/dados_fem_nx4096.npz'
    }

    arquivos_pinn = {
        'PINN [1, 1]': 'npz/dados_pinn_[1, 1, 1].npz',
        'PINN [2, 1]': 'npz/dados_pinn_[1, 2, 1].npz',
        'PINN [5, 1]': 'npz/dados_pinn_[1, 5, 1].npz',
        'PINN [5, 5, 1]': 'npz/dados_pinn_[1, 5, 5, 1].npz',
        'PINN [5, 5, 5, 1]': 'npz/dados_pinn_[1, 5, 5, 5, 1].npz',
        'PINN [10, 1]': 'npz/dados_pinn_[1, 10, 1].npz',
        'PINN [10, 10, 1]': 'npz/dados_pinn_[1, 10, 10, 1].npz',
        'PINN [10, 10, 10, 1]': 'npz/dados_pinn_[1, 10, 10, 10, 1].npz',
        'PINN [20, 1]': 'npz/dados_pinn_[1, 20, 1].npz',
        'PINN [20, 20, 1]': 'npz/dados_pinn_[1, 20, 20, 1].npz',
        'PINN [20, 20, 20, 1]': 'npz/dados_pinn_[1, 20, 20, 20, 1].npz',
        'PINN [40, 1]': 'npz/dados_pinn_[1, 40, 1].npz',
        'PINN [40, 40, 1]': 'npz/dados_pinn_[1, 40, 40, 1].npz',
        'PINN [40, 40, 40, 1]': 'npz/dados_pinn_[1, 40, 40, 40, 1].npz'
    }

    
    # Chama a função de plotagem (opcionalmente passando um caminho para salvar a imagem)
    plotar_comparativo(
        arquivos_pinn=arquivos_pinn,
        arquivos_fem=arquivos_fem,
        x_teste=x_test, 
        caminho_salvar='comparativo_global_poisson.png'
    )

if __name__ == "__main__":
    main()