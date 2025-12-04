# --- teste_modelo_D_super_avancado.py ---
# -*- coding: utf-8 -*-
"""
MODELO D: Super Avançado (Quantum Tensor Networks & Causal/Topological Analysis)
--------------------------------------------------------------------------------
Arquivo de Teste Unitário - Integrante do Framework QPMF.
Atualizado com Metodologias de Fronteira (2025).

Configuração Técnica:
1. Arquitetura: Matrix Product State (MPS) / Quantum Tensor Network.
   - Eficiência de parâmetros e captura de correlações locais (biológicas).
   -Amplitude MPS (StronglyEntangling com topologia vizinho-a-vizinho).
2. Análise Causal: Entropia de Von Neumann.
   - Mede a "complexidade quântica" dos estados de Doença vs Saudável.
3. Análise Topológica: Persistent Homology Proxy.
   - Usa distâncias quânticas para inferir a forma dos dados (Clusters/Buracos).
4. Auditoria: Monitoramento de recursos para justificar o custo computacional.

"""

import sys
import os
import time
import psutil
import datetime
import traceback
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.templates import AmplitudeEmbedding, StronglyEntanglingLayers
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.utils import resample

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
print(">>> INICIANDO TESTE D (MPS BALANCED)")
print(">>> Config: Pesos Moderados (1.2x) | Threshold 0.45 | LR 0.01")
print("█"*80)

# --- CONFIGURAÇÃO DE HARDWARE ---
# Amplitude Embedding comprime 32 features em log2(32) = 5 Qubits
N_QUBITS = 5 
dev = qml.device("default.qubit", wires=N_QUBITS)

# ==============================================================================
# 1. CIRCUITO QUÂNTICO (MPS VENCEDOR)
# ==============================================================================

@qml.qnode(dev)
def qnode_mps_amplitude(inputs, weights):
    """
    Arquitetura Vencedora: Amplitude MPS.
    - Codificação: Amplitude (Densidade Máxima).
    - Ansatz: StronglyEntangling restringido a vizinhos (MPS).
    """
    # 1. Embedding (32 features -> 5 qubits)
    AmplitudeEmbedding(features=inputs, wires=range(N_QUBITS), pad_with=0.0, normalize=True)
    
    # 2. Ansatz MPS (Matrix Product State)
    # ranges=[1]*L garante que cada qubit só fale com seu vizinho imediato (i, i+1).
    n_layers = weights.shape[0]
    StronglyEntanglingLayers(weights, wires=range(N_QUBITS), ranges=[1]*n_layers, imprimitive=qml.CNOT)
    
    return qml.probs(wires=0)

# ==============================================================================
# 2. FUNÇÕES AUXILIARES (PERDA PONDERADA E DIAGNÓSTICO)
# ==============================================================================

def weighted_binary_cross_entropy(probs, y_true, class_weights):
    """Função de custo ponderada para forçar aprendizado da classe minoritária."""
    loss = 0.0
    eps = 1e-7
    for p, y in zip(probs, y_true):
        p_clipped = pnp.clip(p[1], eps, 1 - eps)
        w = class_weights[int(y)]
        term = -(y * pnp.log(p_clipped) + (1 - y) * pnp.log(1 - p_clipped))
        loss += w * term
    return loss / len(y_true)

@qml.qnode(dev)
def density_matrix_circuit(params, x_input):
    """Circuito auxiliar para extrair a Matriz de Densidade (Diagnóstico Causal)."""
    AmplitudeEmbedding(features=x_input, wires=range(N_QUBITS), pad_with=0.0, normalize=True)
    n_layers = params.shape[0]
    StronglyEntanglingLayers(params, wires=range(N_QUBITS), ranges=[1]*n_layers, imprimitive=qml.CNOT)
    return qml.density_matrix(wires=range(N_QUBITS))

def calcular_entropia_von_neumann(rho):
    """S(rho) = -Tr(rho log rho). Mede a incerteza/emaranhamento do estado."""
    try:
        eigvals = np.linalg.eigvalsh(rho)
        eigvals = eigvals[eigvals > 1e-10] # Remover zeros numéricos para o log
        return -np.sum(eigvals * np.log(eigvals))
    except:
        return 0.0

# ==============================================================================
# 3. PIPELINE DE TREINO OTIMIZADO (BALANCED RECALL)
# ==============================================================================

def run_mps_training(X, y, n_layers=5, n_epochs=40):
    """Executa o treino da estratégia com foco em equilíbrio Acurácia/Recall."""
    print(f"\n--- Treinando MPS Amplitude (Modo Balanced) ---")
    print(f"   Features: {X.shape[1]} | Qubits: {N_QUBITS} | Camadas MPS: {n_layers}")
    
    # Split Estratificado
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEED_GLOBAL, stratify=y)
    
    # Pesos de Classe (MODERADO)
    n_pos = np.sum(y_tr == 1)
    w1 = (len(y_tr) / n_pos) if n_pos > 0 else 1.0
    # MULTIPLICADOR 1.2x: Prioriza doença, mas evita destruir a precisão nos saudáveis
    class_weights = {0: 1.0, 1: w1 * 1.2} 
    print(f"   [CONFIG] Pesos de Classe: Neg=1.0, Pos={class_weights[1]:.2f}")
    
    # Threshold de Decisão (AJUSTE FINO)
    # 0.45: Meio termo entre 0.40 (Recall puro) e 0.50 (Acc pura)
    THRESHOLD = 0.45
    print(f"   [CONFIG] Threshold de Decisão: {THRESHOLD} (Equilíbrio)")
    
    # Inicialização "Cold Start"
    shape = StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=N_QUBITS)
    params = pnp.random.normal(0, 0.01, size=shape, requires_grad=True)
    
    # Otimizador Adam com LR reduzido para convergência estável
    opt = qml.AdamOptimizer(0.01)
    
    history_loss = []
    history_acc = []
    history_recall = []
    
    try:
        for ep in range(n_epochs):
            idx = np.random.choice(len(X_tr), 32) 
            X_b, y_b = X_tr[idx], y_tr[idx]
            
            def cost(w):
                probs = [qnode_mps_amplitude(x, w) for x in X_b]
                return weighted_binary_cross_entropy(probs, y_b, class_weights)
                
            params, loss = opt.step_and_cost(cost, params)
            history_loss.append(loss)
            
            # --- MONITORAMENTO ---
            probs_te = np.array([qnode_mps_amplitude(x, params)[1] for x in X_te])
            # Aplicar Threshold Ajustado
            preds_te = np.where(probs_te > THRESHOLD, 1, 0)
            
            acc_ep = balanced_accuracy_score(y_te, preds_te)
            report_ep = classification_report(y_te, preds_te, output_dict=True, zero_division=0)
            recall_ep = report_ep['1']['recall']
            
            history_acc.append(acc_ep)
            history_recall.append(recall_ep)
            
            if (ep+1) % 5 == 0:
                print(f"   Ep {ep+1:02d}: Loss {loss:.4f} | Val Acc: {acc_ep:.2%} | Val Recall: {recall_ep:.2%}")
                
    except Exception as e:
        print(f"   [ERRO CRÍTICO NO TREINO] {str(e)}")
        traceback.print_exc()
        return None

    # Avaliação Final Consolidada
    print("\n   [TREINO CONCLUÍDO] Gerando métricas finais...")
    probs_final = np.array([qnode_mps_amplitude(x, params)[1] for x in X_te])
    preds_final = np.where(probs_final > THRESHOLD, 1, 0) # Threshold 0.45
    
    acc = balanced_accuracy_score(y_te, preds_final)
    cm = confusion_matrix(y_te, preds_final)
    report = classification_report(y_te, preds_final, output_dict=True, zero_division=0)
    
    print(f"   -> Acurácia Balanceada Final: {acc:.2%}")
    print(f"   -> Recall Final (Doença): {report['1']['recall']:.2%}")
    
    return {
        'name': 'Amplitude_MPS_Balanced', 
        'params': params, 
        'acc': acc, 
        'recall': report['1']['recall'], 
        'cm': cm, 
        'loss': history_loss, 
        'history_acc': history_acc, 
        'history_recall': history_recall,
        'X_te': X_te, 'y_te': y_te, 'preds': preds_final
    }

# ==============================================================================
# 4. ANÁLISE CAUSAL E TOPOLÓGICA (DIAGNÓSTICO)
# ==============================================================================

def diagnostico_avancado(result):
    if result is None: return None, 0
    
    print(f"\n[DIAGNÓSTICO AVANÇADO] Investigando Gargalos: {result['name']}")
    
    X_te = result['X_te']
    y_te = result['y_te']
    preds = result['preds']
    params = result['params']
    
    # 1. Análise Causal (Entropia) - Falsos Negativos vs Verdadeiros Positivos
    fn_idx = np.where((y_te==1) & (preds==0))[0]
    tp_idx = np.where((y_te==1) & (preds==1))[0]
    
    entropies_fn = []
    entropies_tp = []
    
    print("   1. Calculando Entropia de Von Neumann (Incerteza Quântica)...")
    limit = 15 
    
    for idx in fn_idx[:limit]:
        rho = density_matrix_circuit(params, X_te[idx])
        entropies_fn.append(calcular_entropia_von_neumann(rho))
        
    for idx in tp_idx[:limit]:
        rho = density_matrix_circuit(params, X_te[idx])
        entropies_tp.append(calcular_entropia_von_neumann(rho))
        
    mean_s_fn = np.mean(entropies_fn) if entropies_fn else 0.0
    mean_s_tp = np.mean(entropies_tp) if entropies_tp else 0.0
    
    print(f"      Entropia Média (Erros/FN): {mean_s_fn:.4f}")
    print(f"      Entropia Média (Acertos/TP): {mean_s_tp:.4f}")
    
    delta_s = mean_s_fn - mean_s_tp
    if delta_s > 0.1:
        print("      >>> DIAGNÓSTICO: 'Confusão Quântica'. Os erros ocorrem em estados de alta entropia.")
    elif delta_s < -0.1:
        print("      >>> DIAGNÓSTICO: 'Incoerência Estrutural'. O modelo está confiante mas errado.")
    else:
        print("      >>> DIAGNÓSTICO: 'Falta de Expressividade'. A entropia não explica o erro.")

    # 2. Análise Topológica (Betti-0 Proxy)
    print("\n   2. Análise Topológica (Conectividade dos Erros)...")
    if len(fn_idx) > 1:
        X_errors = X_te[fn_idx]
        dist_matrix = np.linalg.norm(X_errors[:, None] - X_errors, axis=2)
        
        threshold = np.mean(dist_matrix) + 1e-9
        adj = (dist_matrix < threshold).astype(int)
        G = nx.from_numpy_array(adj)
        n_comp = nx.number_connected_components(G)
        
        print(f"      Componentes Conectados (Betti-0 Proxy) nos Erros: {n_comp}")
        return G, n_comp
    else:
        print("      (Poucos erros para análise topológica confiável)")
        return None, 0

# ==============================================================================
# 5. MAIN
# ==============================================================================

def executar_teste():
    monitor = ResourceMonitor("Teste_D_MPS_Balanced")
    monitor.start()
    
    # Carregar Dados
    X, y = carregar_dados_benchmark(tipo_scaling='quantum')
    
    # Executar (5 Camadas, 50 Épocas para estabilidade)
    resultado = run_mps_training(X, y, n_layers=5, n_epochs=50)
    
    if resultado:
        # Diagnóstico
        G_topo, n_comp = diagnostico_avancado(resultado)
        
        # Salvar Resultados
        df = pd.DataFrame([{
            'Modelo': resultado['name'],
            'Acuracia': resultado['acc'],
            'Recall_Doenca': resultado['recall'],
            'Topologia_Betti0_Erros': n_comp
        }])
        df.to_csv(os.path.join(PASTA_RESULTADOS, 'resultado_modelo_D_final.csv'), index=False)
        
        # Gráficos
        try:
            print("\n[GRÁFICOS] Gerando painel de controle visual...")
            
            plt.figure(figsize=(10, 12))
            
            plt.subplot(3, 1, 1)
            plt.plot(resultado['loss'], label='Perda (Loss)', color='purple', marker='o')
            plt.title("Otimização MPS (Balanced Recall)")
            plt.ylabel("Weighted Loss")
            plt.grid(True, alpha=0.3)
            
            plt.subplot(3, 1, 2)
            plt.plot(resultado['history_acc'], label='Acurácia', color='green')
            plt.ylabel("Balanced Acc")
            plt.grid(True, alpha=0.3)
            
            plt.subplot(3, 1, 3)
            plt.plot(resultado['history_recall'], label='Recall Doença', color='red', linewidth=2)
            plt.ylabel("Recall")
            plt.xlabel("Época")
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(PASTA_GRAFICOS, 'modelo_D_metricas_balanced.png'))
            plt.close()
            
            if G_topo:
                plt.figure(figsize=(8, 8))
                pos = nx.spring_layout(G_topo, seed=42)
                nx.draw(G_topo, pos, node_color='crimson', node_size=120, alpha=0.7)
                plt.title(f"Topologia dos Erros\nClusters: {n_comp}")
                plt.savefig(os.path.join(PASTA_GRAFICOS, 'modelo_D_topologia_erros.png'))
                plt.close()
                
            print(f"[SUCESSO] Gráficos salvos em {PASTA_GRAFICOS}")
            
        except Exception as e:
            print(f"[AVISO] Erro nos gráficos: {e}")
            
    monitor.stop()

if __name__ == "__main__":
    executar_teste()