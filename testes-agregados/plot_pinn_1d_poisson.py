import json
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_results():
    # Definição das arquiteturas testadas
    architectures = [
        # [1, 1], [2, 1],
        [5, 1], [10, 1], [20, 1], [40, 1],
        [5, 5, 1], [10, 10, 1], [20, 20, 1], [40, 40, 1],
        [5, 5, 5, 1], [10, 10, 10, 1], [20, 20, 20, 1], [40, 40, 40, 1]
    ]

    # Configuração da figura
    fig, ax = plt.subplots(figsize=(8, 6))

    # 1. Carregar e plotar a solução analítica (Ground Truth)
    gt_filename = "gt_poisson_1d.json"
    try:
        with open(gt_filename, 'r') as f:
            gt_data = json.load(f)
        
        # Extrair e achatar (flatten) os dados da solução analítica gerada pelo script fornecido
        x_gt = np.array(gt_data["X"]).flatten()
        u_gt = np.array(gt_data["U_true"]).flatten()
        
        # Plotar a linha do Ground Truth
        ax.plot(x_gt, u_gt, color='red', linewidth=3.0, label='Ground Truth solution')
        
        # Adicionar os marcadores 'x' azuis nas bordas (x=0 e x=1) para seguir o estilo da imagem original
        ax.plot([x_gt[0], x_gt[-1]], [u_gt[0], u_gt[-1]], 'bx', markersize=8)
        
    except FileNotFoundError:
        print(f"Aviso: Arquivo {gt_filename} não encontrado. Execute o ground_truth_poisson_1d.py primeiro.")

    # 2. Iterar sobre as arquiteturas, carregar os JSONs correspondentes e plotar
    for arch in architectures:
        width = [1] + arch
        arch_str = "_".join(map(str, width))
        nome_arquivo = f'pinn_poisson_1d/pontos_pinn_1d_{arch_str}.json'
        
        if os.path.exists(nome_arquivo):
            with open(nome_arquivo, 'r') as f:
                pinn_data = json.load(f)
                
            x_pinn = pinn_data['x']
            y_pinn = pinn_data['y_nn']
            
            # Formatando o label no estilo PINN [camadas]
            label_name = f"PINN {arch}"
            
            # Plotar a aproximação usando linhas mais finas (linewidth=1.2)
            ax.plot(x_pinn, y_pinn, linewidth=1.2, label=label_name)
        else:
            print(f"Arquivo não encontrado: {nome_arquivo}")

    # 3. Formatação do gráfico no estilo da imagem 'image_ac7ac6.png'
    ax.set_xlabel('x')
    ax.set_ylabel('u(x)')
    
    # Posicionar a legenda fora do gráfico, alinhada à direita
    # bbox_to_anchor move a legenda para fora da área dos eixos
    ax.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0.)
    
    # Ajustar o layout para garantir que a legenda caiba na imagem salva
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    # Salvar e mostrar o gráfico
    output_img = 'POISSON-1D-PINN-COMPARATIVO.png'
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Gráfico gerado e salvo como: {output_img}")
    plt.show()

if __name__ == "__main__":
    plot_results()