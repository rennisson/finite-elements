import json
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_fem_results():
    # Tamanhos de malha (nx) testados no script fem_poisson_1d.py
    mesh_sizes = [64, 128, 256, 512, 1024, 2048, 4096]

    # Configuração da figura
    fig, ax = plt.subplots(figsize=(8, 6))

    # 1. Carregar e plotar a solução analítica (Ground Truth)
    gt_filename = "gt_poisson_1d.json"
    try:
        with open(gt_filename, 'r') as f:
            gt_data = json.load(f)
        
        # Extrair e achatar (flatten) os dados da solução analítica
        x_gt = np.array(gt_data["X"]).flatten()
        u_gt = np.array(gt_data["U_true"]).flatten()
        
        # Plotar a linha do Ground Truth
        ax.plot(x_gt, u_gt, color='red', linewidth=3.0, label='Ground Truth solution')
        
        # Adicionar marcadores 'x' azuis nas bordas (x=0 e x=1)
        ax.plot([x_gt[0], x_gt[-1]], [u_gt[0], u_gt[-1]], 'bx', markersize=8)
        
    except FileNotFoundError:
        print(f"Aviso: Arquivo {gt_filename} não encontrado.")

    # 2. Iterar sobre as resoluções de malha FEM e plotar as curvas
    for nx in mesh_sizes:
        nome_arquivo = f'fem_poisson_1d/dados_fem_nx{nx}.json'
        
        if os.path.exists(nome_arquivo):
            with open(nome_arquivo, 'r') as f:
                fem_data = json.load(f)
                
            x_fem = np.array(fem_data['x_fem']).flatten()
            y_fem = np.array(fem_data['y_fem']).flatten()
            
            # Rotulação no gráfico especificando o número de divisões da malha
            label_name = f"FEM (nx={nx})"
            
            # Plotar a aproximação FEM
            ax.plot(x_fem, y_fem, linewidth=1.2, label=label_name)
        else:
            print(f"Arquivo não encontrado: {nome_arquivo}")

    # 3. Formatação do gráfico no estilo do plot_1d_poisson original
    ax.set_xlabel('x')
    ax.set_ylabel('u(x)')
    
    # Posicionar a legenda fora da área dos eixos (alinhada à direita)
    ax.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0.)
    
    # Ajustar o layout para evitar corte da legenda
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    # Salvar e exibir o gráfico
    output_img = 'POISSON-1D-FEM-COMPARATIVO.png'
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Gráfico gerado e salvo como: {output_img}")
    plt.show()

if __name__ == "__main__":
    plot_fem_results()