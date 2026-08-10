import os
import subprocess
import sys

def run_script(script_path):
    print(f"\n=========================================")
    print(f"Executando: {script_path}")
    print(f"=========================================\n")
    
    if not os.path.exists(script_path):
        print(f"ERRO: Arquivo {script_path} não encontrado!")
        return False
        
    try:
        result = subprocess.run([sys.executable, script_path], check=True)
        print(f"\n✅ Sucesso: {script_path} concluído.\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERRO ao executar {script_path}. Código de saída: {e.returncode}")
        return False

def main():
    scripts_to_run = [
        "fem/fem_poisson_1d.py",
        "pinns/pinn_poisson_1d.py",
        "sv-pinns/svpinn_poisson_1d.py"
    ]
    
    print("Iniciando Pipeline de Execução Poisson 1D...")
    
    for script in scripts_to_run:
        success = run_script(script)
        if not success:
            print("Pipeline interrompido devido a erro.")
            break
            
    print("Pipeline finalizado.")

if __name__ == "__main__":
    main()