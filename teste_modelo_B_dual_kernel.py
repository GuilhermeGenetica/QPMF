# --- teste_modelo_B_dual_kernel.py ---
# -*- coding: utf-8 -*-
"""
MODELO B: Dual Kernel SVM (Híbrido) + QSVM Otimizado (Angle Kernel 8Q)
----------------------------------------------------------------------
Arquivo de Teste Unitário - Integrante do Framework QPMF.
Atualizado com Auditoria de Recursos e Análise de Divergência de Kernel (2025).

ATUALIZAÇÃO  (Estratégia 8 Qubits):
- Diagnóstico Anterior: QSVM Angle 4Q (77.8%) foi excelente, mas limitado pela baixa
  variância retida (~19%) do PCA.
- Ajuste: Aumentar para 8 Qubits (PCA 8 Componentes).
- Objetivo: Reter ~60-70% da informação para tentar superar o RBF Clássico (80%).

Configuração Técnica Original:
- Kernel Quântico 1: Amplitude Embedding (Overlap/Fidelity Kernel).
- Kernel Clássico: RBF (Radial Basis Function).
- Estratégia: Combinação Convexa (Alpha * K_Q + (1-Alpha) * K_C).

Configuração Técnica Extra (QSVM "Sweet Spot"):
- Kernel Quântico 2: Angle Embedding (Cosine/Rotation Kernel).
- Pre-processamento: PCA (32 -> 8 Features) + Scaling [0, π].
- Hardware: 8 Qubits (Equilíbrio ideal Informação/Ruído).
"""

import sys
import os
import time
import psutil
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pennylane as qml
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import StratifiedKFold
# Adições para a estratégia otimizada
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Tenta importar codecarbon para rastreio local
try:
    from codecarbon import EmissionsTracker
    HAS_CODECARBON = True
except ImportError:
    HAS_CODECARBON = False

# Importação do Config Comum
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config_comum import *

# --- CLASSE DE MONITORAMENTO ---
class ResourceMonitor:
    def __init__(self, task_name):
        self.task_name = task_name
        self.tracker = EmissionsTracker(project_name=task_name, log_level='error') if HAS_CODECARBON else None
        self.start_time = None
        
    def start(self):
        self.start_time = time.time()
        print(f"[{datetime.datetime.now()}] Iniciando tarefa: {self.task_name}")
        print(f"   Recursos Iniciais: CPU {psutil.cpu_percent()}% | RAM {psutil.virtual_memory().percent}%")
        if self.tracker: self.tracker.start()
        
    def stop(self):
        duration = time.time() - self.start_time
        emissions = 0.0
        if self.tracker:
            try: emissions = self.tracker.stop()
            except: pass
        
        print(f"[{datetime.datetime.now()}] Tarefa Finalizada: {self.task_name}")
        print(f"   Duração: {duration:.2f}s | Emissões: {emissions:.6f} kgCO2eq")
        print(f"   Recursos Finais: CPU {psutil.cpu_percent()}% | RAM {psutil.virtual_memory().percent}%")
        return duration, emissions

print("\n" + "█"*80)
print(">>> INICIANDO TESTE B: DUAL KERNEL + QSVM ANGLE 8-QUBITS (SWEET SPOT)")
print(">>> Config: Grid Search Alpha (Dual) | PCA 8 (Angle)")
print("█"*80)

# ==============================================================================
# 1. ARQUITETURA ORIGINAL (AMPLITUDE KERNEL)
# ==============================================================================
WIRES_DATA = range(5) # 2^5 = 32 features
dev = qml.device("default.qubit", wires=N_WIRES)

@qml.qnode(dev)
def kernel_circuit(x1, x2):
    """
    Calcula o overlap (fidelidade) entre dois estados de dados codificados.
    Amplitude Embedding mapeia vetor de 32 features -> Estado de 5 qubits.
    |<psi(x1)|psi(x2)>|^2
    """
    qml.AmplitudeEmbedding(features=x1, wires=WIRES_DATA, pad_with=0.0, normalize=True)
    qml.adjoint(qml.AmplitudeEmbedding)(features=x2, wires=WIRES_DATA, pad_with=0.0, normalize=True)
    return qml.probs(wires=WIRES_DATA)

def compute_element(x1, x2):
    # A probabilidade do estado |00000> após U(x1)U^dag(x2) é a fidelidade.
    return kernel_circuit(x1, x2)[0]

def get_kernel_matrix(X):
    """
    Gerencia o cálculo ou carregamento da Matriz de Kernel (Gram Matrix).
    Implementa tolerância a falhas no carregamento do cache.
    """
    # 1. Tentativa de Cache
    if os.path.exists(ARQUIVO_CACHE_KERNEL):
        try:
            print("  [IO] Tentando carregar cache global de Kernel (Amplitude)...")
            K = np.load(ARQUIVO_CACHE_KERNEL)
            if K.shape[0] == len(X):
                print("  [CACHE] Matriz Kernel Global carregada e validada.")
                return np.nan_to_num(K)
            else:
                print(f"  [AVISO] Cache dimensionalmente incompatível ({K.shape[0]} vs {len(X)}). Recalculando.")
        except Exception as e:
            print(f"  [ERRO] Cache corrompido: {e}. Recalculando.")
    
    # 2. Cálculo (Se cache falhar ou não existir)
    print("  [PROCESSAMENTO] Calculando Kernel Quântico Amplitude (O(N^2))...")
    N = len(X)
    K = np.zeros((N, N))
    
    start_k = time.time()
    # Otimização: Matriz é simétrica, calculamos apenas triângulo superior
    count = 0
    total_ops = N * (N + 1) // 2
    
    for i in range(N):
        for j in range(i, N):
            v = compute_element(X[i], X[j])
            K[i,j] = K[j,i] = v
            count += 1
            if count % 5000 == 0:
                elapsed = time.time() - start_k
                eta = (elapsed / count) * (total_ops - count)
                print(f"    Progresso: {count/total_ops:.1%} | ETA: {eta:.0f}s", end='\r')
                
    print(f"\n  [CONCLUÍDO] Cálculo finalizado em {time.time() - start_k:.1f}s")
    
    # Salvar novo cache
    try:
        np.save(ARQUIVO_CACHE_KERNEL, K)
        print("  [IO] Cache salvo em disco.")
    except:
        print("  [AVISO] Não foi possível salvar o cache.")
        
    return np.nan_to_num(K)

# ==============================================================================
# 2. ARQUITETURA EXTRA OTIMIZADA (ANGLE KERNEL 8 QUBITS)
# ==============================================================================
# : Aumentado para 8 Qubits para capturar mais variância
N_WIRES_ANGLE = 8
dev_angle = qml.device("default.qubit", wires=N_WIRES_ANGLE)
CACHE_ANGLE = os.path.join(PASTA_RESULTADOS, "cache_kernel_angle_8q.npy")

@qml.qnode(dev_angle)
def kernel_circuit_angle(x1, x2):
    """
    Kernel Geométrico em 8 Qubits.
    Permite espaço de Hilbert de 2^8 = 256 dimensões.
    """
    qml.AngleEmbedding(features=x1, wires=range(N_WIRES_ANGLE), rotation='Y')
    qml.adjoint(qml.AngleEmbedding)(features=x2, wires=range(N_WIRES_ANGLE), rotation='Y')
    return qml.probs(wires=range(N_WIRES_ANGLE))

def compute_element_angle(x1, x2):
    return kernel_circuit_angle(x1, x2)[0]

def get_kernel_matrix_angle_optimized(X_pca):
    """
    Calcula matriz para a versão otimizada (Angle Encoding 8Q).
    Usa cache separado para não conflitar.
    """
    if os.path.exists(CACHE_ANGLE):
        try:
            print("  [IO] Tentando carregar cache Kernel Otimizado (Angle 8Q)...")
            K = np.load(CACHE_ANGLE)
            if K.shape[0] == len(X_pca):
                print("  [CACHE] Matriz Angle Kernel 8Q carregada.")
                return np.nan_to_num(K)
        except: pass

    print(f"  [PROCESSAMENTO] Calculando Kernel Angle Otimizado ({N_WIRES_ANGLE} Qubits)...")
    N = len(X_pca)
    K = np.zeros((N, N))
    start_k = time.time()
    count = 0
    total_ops = N * (N + 1) // 2
    
    for i in range(N):
        for j in range(i, N):
            v = compute_element_angle(X_pca[i], X_pca[j])
            K[i,j] = K[j,i] = v
            count += 1
            if count % 2000 == 0: # Log mais frequente
                elapsed = time.time() - start_k
                eta = (elapsed / (count+1)) * (total_ops - count)
                print(f"    Progresso Angle: {count/total_ops:.1%} | ETA: {eta:.0f}s", end='\r')
                
    print(f"\n  [CONCLUÍDO] Kernel Angle calculado em {time.time() - start_k:.1f}s")
    try: np.save(CACHE_ANGLE, K)
    except: pass
    return np.nan_to_num(K)

# ==============================================================================
# 3. EXECUÇÃO PRINCIPAL
# ==============================================================================

def executar_teste():
    monitor = ResourceMonitor("Teste_B_DualKernel_Completo")
    monitor.start()
    
    # Carregar Dados Originais (Quantum Scaling para o teste original)
    X, y = carregar_dados_benchmark(tipo_scaling='quantum')
    
    # ---------------------------------------------------------
    # PARTE A: DUAL KERNEL ORIGINAL (AMPLITUDE)
    # ---------------------------------------------------------
    print("\n--- 1. Preparação de Kernels (Original) ---")
    K_quantico = get_kernel_matrix(X)
    
    print("  [CLASSICO] Calculando Kernel RBF (Referência)...")
    # RBF clássico para comparação e hibridização
    K_classico = rbf_kernel(X, X)
    
    # Análise de Divergência (Quantumness)
    # Mede o quão diferente a visão quântica é da clássica (Norma de Frobenius da diferença)
    diff_norm = np.linalg.norm(K_quantico - K_classico)
    print(f"  [ANÁLISE] Divergência Quantum-Classic (Frobenius): {diff_norm:.4f}")
    
    # 2. Otimização Alpha
    alphas = [0.0, 0.05, 0.1, 0.15, 0.2, 1.0] # 0.0 = Puro Clássico, 1.0 = Puro Quântico
    
    # Para plot
    alpha_plot = []
    score_plot = []
    
    # K-Fold fixo para comparação justa no Grid Search
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED_GLOBAL) 
    
    print("\n--- 2. Otimização de Mistura (Grid Search Alpha) ---")
    
    best_avg_acc = -1.0
    best_alpha = -1.0
    best_std = 0.0
    
    for alpha in alphas:
        # Matriz Combinada: K_mix = αK_Q + (1-α)K_C
        K_mix_global = alpha * K_quantico + (1 - alpha) * K_classico
        
        fold_scores = []
        
        for tr_idx, te_idx in skf.split(X, y):
            # Fatiamento da Matriz de Kernel pré-calculada (Truque do Kernel Trick)
            K_train = K_mix_global[np.ix_(tr_idx, tr_idx)]
            K_test = K_mix_global[np.ix_(te_idx, tr_idx)]
            y_train, y_test = y[tr_idx], y[te_idx]
            
            svm = SVC(kernel='precomputed', class_weight='balanced', cache_size=1000)
            svm.fit(K_train, y_train)
            
            pred = svm.predict(K_test)
            score = balanced_accuracy_score(y_test, pred)
            fold_scores.append(score)
            
        avg = np.mean(fold_scores)
        std = np.std(fold_scores)
        
        alpha_plot.append(alpha)
        score_plot.append(avg)
        
        tipo = "MISTO"
        if alpha == 0.0: tipo = "CLÁSSICO (RBF)"
        if alpha == 1.0: tipo = "QUÂNTICO (AMP)"
        
        print(f"   Alpha {alpha:.1f} [{tipo:^15}]: Bal. Acc = {avg:.2%} (+/- {std:.2%})")
        
        if avg > best_avg_acc:
            best_avg_acc = avg
            best_alpha = alpha
            best_std = std
            
    print("\n" + "-"*40)
    print(f"[RESULTADO PARCIAL MODELO B - DUAL]")
    print(f"Melhor Configuração: Alpha = {best_alpha}")
    print(f"Média Acurácia Balanceada: {best_avg_acc:.2%} (+/- {best_std:.2%})")
    print("-" * 40)
    
    # ---------------------------------------------------------
    # PARTE B: QSVM EXTRA OTIMIZADA (ANGLE + PCA 8Q)
    # ---------------------------------------------------------
    print("\n" + "█"*80)
    print(f">>> 3. TESTE ISOLADO: QSVM OTIMIZADA (ANGLE KERNEL + PCA {N_WIRES_ANGLE}Q)")
    print(">>> Estratégia 'Sweet Spot': Maior Variância, Ruído Gerível")
    print("█"*80)

    # 1. Pipeline de Dados Otimizado (Classic Scaling -> PCA -> Angle Scaling)
    print("   [PREP] Preparando dados para Angle Kernel...")
    X_raw, _ = carregar_dados_benchmark(tipo_scaling='classic') # Usa raw para PCA correto
    
    # PCA para 8 componentes
    pca = PCA(n_components=N_WIRES_ANGLE, random_state=SEED_GLOBAL)
    X_pca = pca.fit_transform(X_raw)
    var_retida = np.sum(pca.explained_variance_ratio_)
    print(f"   [PCA] Redução 32 -> {N_WIRES_ANGLE} Dimensões (Variância: {var_retida:.2%})")
    
    # Scaling para [0, PI] (Crucial para Angle Embedding)
    scaler_angle = MinMaxScaler(feature_range=(0, np.pi))
    X_angle_ready = scaler_angle.fit_transform(X_pca)
    
    # 2. Calcular Kernel Geométrico
    K_angle = get_kernel_matrix_angle_optimized(X_angle_ready)
    
    # 3. Avaliação Cross-Validation (Mesmo Split)
    angle_scores = []
    print(f"\n   [AVALIAÇÃO] Rodando CV na QSVM Angle {N_WIRES_ANGLE}Q...")
    
    for tr_idx, te_idx in skf.split(X, y):
        K_tr = K_angle[np.ix_(tr_idx, tr_idx)]
        K_te = K_angle[np.ix_(te_idx, tr_idx)]
        y_tr, y_te = y[tr_idx], y[te_idx]
        
        # SVM Puro com Kernel Quântico Otimizado
        svm_angle = SVC(kernel='precomputed', class_weight='balanced')
        svm_angle.fit(K_tr, y_tr)
        pred_angle = svm_angle.predict(K_te)
        
        scr = balanced_accuracy_score(y_te, pred_angle)
        angle_scores.append(scr)
        print(f"    -> Fold Acc: {scr:.2%}")

    avg_angle = np.mean(angle_scores)
    std_angle = np.std(angle_scores)

    print("\n" + "-"*40)
    print(f"[RESULTADO FINAL - COMPARAÇÃO]")
    print(f"1. Dual Kernel (Best Alpha={best_alpha}): {best_avg_acc:.2%}")
    print(f"2. QSVM Angle {N_WIRES_ANGLE}Q (Pura):         {avg_angle:.2%} (+/- {std_angle:.2%})")
    
    if avg_angle > best_avg_acc:
        print(">>> CONCLUSÃO: A estratégia Angle 8Q SUPEROU o Clássico/Dual!")
    else:
        print(">>> CONCLUSÃO: O Dual/Clássico ainda lidera, mas o gap reduziu?")
    print("-" * 40)

    monitor.stop()
    
    # Salvar Resultados
    # 1. Resultado para o Benchmark Master
    df = pd.DataFrame([{
        'Modelo': f'Dual_Kernel_Alpha_{best_alpha}', 
        'Acuracia_Media': best_avg_acc, 
        'Std_Dev': best_std,
        'Alpha_Otimo': best_alpha,
        'Divergencia_Kernel': diff_norm
    }])
    df.to_csv(os.path.join(PASTA_RESULTADOS, 'resultado_modelo_B.csv'), index=False)
    
    # 2. Resultado Extra
    df_extra = pd.DataFrame([{
        'Modelo': f'QSVM_Angle_{N_WIRES_ANGLE}Q',
        'Acuracia_Media': avg_angle,
        'Std_Dev': std_angle,
        'Features': N_WIRES_ANGLE,
        'Variancia_Retida': var_retida
    }])
    df_extra.to_csv(os.path.join(PASTA_RESULTADOS, 'resultado_modelo_B_extra_qsvm.csv'), index=False)

    # --- GERAÇÃO DE GRÁFICOS ---
    try:
        print("\n[GRÁFICOS] Gerando curvas comparativas Modelo B...")
        plt.figure(figsize=(10, 6))
        
        plt.plot(alpha_plot, score_plot, marker='o', linewidth=2, color='darkgreen', label='Dual Kernel (Hybrid)')
        plt.axvline(best_alpha, color='red', linestyle='--', alpha=0.5, label=f'Melhor Alpha Dual: {best_alpha}')
        
        plt.axhline(avg_angle, color='blue', linestyle='-.', linewidth=2, label=f'QSVM Angle {N_WIRES_ANGLE}Q: {avg_angle:.2%}')
        
        plt.title(f'Batalha de Kernels: Dual Híbrido vs QSVM Angle {N_WIRES_ANGLE}Q')
        plt.xlabel('Alpha (0=RBF Clássico, 1=Quantum Amplitude)')
        plt.ylabel('Acurácia Balanceada')
        plt.xticks(alphas)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.savefig(os.path.join(PASTA_GRAFICOS, 'modelo_B_comparacao_kernels.png'))
        plt.close()
        print(f"[SUCESSO] Gráfico salvo em {PASTA_GRAFICOS}")
    except Exception as e:
        print(f"[AVISO] Erro ao gerar gráficos: {e}")

if __name__ == "__main__":
    executar_teste()