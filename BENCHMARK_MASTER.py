# --- BENCHMARK_MASTER.py ---
# -*- coding: utf-8 -*-
"""
BENCHMARK MASTER - FRAMEWORK QPMF (Suite Completa & Granular - Otimizada 2025)
------------------------------------------------------------------------------
Orquestrador Mestre de Auditoria e Execução.
Atualizado para padrões científicos de 2025:
- Monitoramento de Recursos (CPU/RAM/Energia Estimada).
- Log de Versionamento para Reprodutibilidade.
- Consolidação Estatística de Resultados.
- Tolerância a Falhas em Subprocessos.

Lista de Execução Atualizada:
1. Clássico Robusto (Baseline)
2. Modelo A (QNN Otimizado)
3. Modelo B (Dual Kernel)
4. Modelo C (Quantum Boosting)
5. Modelo D (Super Avançado - Entropia/Topologia)
6. Modelo E (Angle Encoding Exploratório)
7. Modelo F (Hierarchical TTN & MERA)
8. Modelo G (QNN ICO - Interference) [NOVO]
9. Suite Quântica Geral (Anti-BP) - Compara A, B, C, D, E, G internamente.
"""

import os
import sys
import time
import pandas as pd
import subprocess
import platform
import psutil
import datetime
import pkg_resources
import numpy as np

# Tenta importar codecarbon para rastreio real de emissões, senão simula
try:
    from codecarbon import EmissionsTracker
    HAS_CODECARBON = True
except ImportError:
    HAS_CODECARBON = False

# --- CONFIGURAÇÃO ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA_RES = os.path.join(BASE_DIR, 'RESULTADOS_QPMF')

# Lista de Pipelines: (Nome Visível, Arquivo Script, Arquivo CSV Gerado)
PIPELINES = [
    # 1. Baseline Clássico (Referência Absoluta)
    ("1. Suite Clássica (Robusta)", "benchmark_classico_robusto.py", "resultados_classicos_completos.csv"),
    
    # 2. Testes Quânticos Individuais (Granularidade Máxima)
    ("2. Modelo A (QNN Otimizado)",  "teste_modelo_A_qnn_otimizado.py",  "resultado_modelo_A.csv"),
    ("3. Modelo B (Dual Kernel)",    "teste_modelo_B_dual_kernel.py",    "resultado_modelo_B.csv"),
    ("4. Modelo C (AdaBoost-QNN)",       "teste_modelo_C_boosting.py",       "resultado_modelo_C.csv"),
    ("5. Modelo D (MPS)", "teste_modelo_D_super_avancado.py", "resultado_modelo_D_final.csv"),   
    ("6. Modelo E (Angle Encoding)", "teste_modelo_E_angle_encoding.py", "resultado_modelo_E.csv"),
    ("7. Modelo F (TTN & MERA)",     "teste_modelo_F_TTN.py",            "resultado_modelo_F_grid.csv"),
    ("8. Modelo G (QNN ICO)",        "teste_modelo_G_qnn_ico.py",        "resultado_modelo_G.csv"),
    ("9. Suite Quantica Geral - TODOS (A-G)",     "benchmark_quantico_anti_bp.py",   "resultados_quanticos_otimizados.csv")
]

LOG_FILE = os.path.join(PASTA_RES, 'LOG_GERAL_BENCHMARK.txt')

def log(msg):
    """Registra mensagens no console e no arquivo de log com timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    print(formatted_msg)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(formatted_msg + "\n")

def check_system_resources():
    """Auditoria inicial do sistema para garantir estabilidade dos testes."""
    log("\n--- AUDITORIA DE SISTEMA ---")
    log(f"OS: {platform.system()} {platform.release()}")
    log(f"Processador: {platform.processor()}")
    log(f"Núcleos Físicos: {psutil.cpu_count(logical=False)}")
    log(f"Memória Total: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    log(f"Memória Disponível: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    
    # Versionamento de Bibliotecas Críticas
    libs = ['pennylane', 'numpy', 'scikit-learn', 'pandas', 'scipy']
    log("Versões de Bibliotecas Críticas:")
    for lib in libs:
        try:
            ver = pkg_resources.get_distribution(lib).version
            log(f"  - {lib}: {ver}")
        except:
            log(f"  - {lib}: Não detectado/Instalação não padrão")

def normalizar_dataframe(df, origem):
    """
    Padroniza nomes de colunas diferentes (Acuracia, Bal_Acc, etc) para um Ranking único.
    Garante integridade dos dados para o Leaderboard.
    """
    df['Origem'] = origem
    
    # Renomear colunas de score para 'Score_Final'
    cols_map = {
        'Acuracia': 'Score_Final',
        'Acuracia_Media': 'Score_Final',
        'Bal_Acc': 'Score_Final',
        'Balanced_Accuracy': 'Score_Final',
        'Mean_Accuracy': 'Score_Final'
    }
    
    df = df.rename(columns=cols_map)
    
    # Garantir colunas essenciais
    if 'Score_Final' not in df.columns:
        # Tenta pegar a primeira coluna numérica como score se não achar pelo nome
        nums = df.select_dtypes(include=[np.number])
        if not nums.empty:
            df['Score_Final'] = nums.iloc[:, 0]
        else:
            df['Score_Final'] = 0.0
            
    if 'Std_Dev' not in df.columns:
        df['Std_Dev'] = 0.0
        
    return df[['Modelo', 'Score_Final', 'Std_Dev', 'Origem']]

def run_benchmark():
    start_global = time.time()
    os.makedirs(PASTA_RES, exist_ok=True)
    
    # Inicializa Log Limpo
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("=== LOG DE EXECUÇÃO - QPMF MASTER (v2025) ===\n\n")
    
    check_system_resources()
    
    # Rastreamento Energético Global
    tracker = None
    if HAS_CODECARBON:
        try:
            tracker = EmissionsTracker(output_dir=PASTA_RES, project_name="QPMF_Benchmark_Global")
            tracker.start()
            log("--- MONITORAMENTO DE ENERGIA (CodeCarbon) ATIVADO ---")
        except Exception as e:
            log(f"[AVISO] Falha ao iniciar CodeCarbon: {e}")
    else:
        log("--- MONITORAMENTO DE ENERGIA: MODO SIMULADO (CodeCarbon ausente) ---")

    results_dfs = []
    
    log("\n" + "█"*80)
    log("INICIANDO A GRANDE BATALHA DE ALGORITMOS (BENCHMARK QUÂNTICO)")
    log("█"*80)
    
    for nome_suite, script_file, output_csv in PIPELINES:
        script_path = os.path.join(BASE_DIR, script_file)
        
        log(f"\n>>> EXECUTANDO: {nome_suite}")
        log(f"    Arquivo: {script_file}")
        
        if not os.path.exists(script_path):
            log(f"    [ERRO] Arquivo não encontrado: {script_file}. Pulando...")
            continue
            
        t0 = time.time()
        
        try:
            # Executa o script e espera terminar.
            # O output do script vai direto para o terminal para acompanhamento em tempo real.
            log(f"    ...Processando {nome_suite}...")
            ret_code = subprocess.call([sys.executable, script_path])
            
            tempo = time.time() - t0
            
            if ret_code == 0:
                log(f"    [SUCESSO] Concluído em {tempo:.2f}s")
                
                # Carregar Resultado
                csv_path = os.path.join(PASTA_RES, output_csv)
                if os.path.exists(csv_path):
                    try:
                        df_raw = pd.read_csv(csv_path)
                        if not df_raw.empty:
                            df_norm = normalizar_dataframe(df_raw, nome_suite)
                            results_dfs.append(df_norm)
                            log(f"    [DADOS] Resultados capturados de {output_csv}")
                        else:
                            log(f"    [AVISO] O CSV {output_csv} está vazio.")
                    except Exception as e:
                        log(f"    [ERRO LEITURA] Falha ao ler CSV: {e}")
                else:
                    log(f"    [AVISO] CSV de saída não gerado: {output_csv}")
            else:
                log(f"    [FALHA] Erro na execução do subprocesso (Código {ret_code})")
                log("    Verifique os logs individuais do teste acima.")
                
        except Exception as e:
            log(f"    [ERRO CRÍTICO] Falha ao lançar subprocesso: {e}")
            
    # Finaliza Rastreio Energético
    emissions = 0.0
    if tracker:
        try:
            emissions = tracker.stop()
            log(f"\n[ENERGIA] Emissões Totais Estimadas: {emissions:.6f} kgCO2eq")
        except: pass

    log("\n" + "█"*80)
    log("RANKING FINAL UNIFICADO (LEADERBOARD)")
    log("█"*80)
    
    if results_dfs:
        df_final = pd.concat(results_dfs, ignore_index=True)
        # Ordenar pelo Score Final (Acurácia Balanceada)
        df_final.sort_values(by='Score_Final', ascending=False, inplace=True)
        df_final.reset_index(drop=True, inplace=True)
        
        # Formatação para exibição bonita
        log("\n" + df_final.to_string(index=True))
        
        caminho_placar = os.path.join(PASTA_RES, 'PLACAR_FINAL_GERAL.csv')
        df_final.to_csv(caminho_placar, index=False)
        log(f"\nPlacar salvo em: {caminho_placar}")
        
        top = df_final.iloc[0]
        log(f"\n[VENCEDOR ATUAL]: {top['Modelo']}")
        log(f"   Score Final: {top['Score_Final']:.2%}")
        log(f"   Origem: {top['Origem']}")
        log(f"   Estabilidade (StdDev): {top['Std_Dev']:.4f}")
    else:
        log("Nenhum resultado válido foi coletado. Verifique os scripts individuais.")
        
    total_time = time.time() - start_global
    log(f"\nTempo Total do Benchmark: {total_time:.2f}s ({total_time/60:.2f} min)")

if __name__ == "__main__":
    run_benchmark()