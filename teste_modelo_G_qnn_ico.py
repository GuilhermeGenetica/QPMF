# --- teste_modelo_G_qnn_ico.py ---
# -*- coding: utf-8 -*-
"""
MODELO G: QNN ICO (Interference & Control Optimization)
-------------------------------------------------------
Arquivo de Teste Unitário - Integrante do Framework QPMF.
Separado com Independência Total (2025).

Descrição Técnica:
Este modelo explora explicitamente a interferência quântica controlada.
Ao contrário do modelo A (Força Bruta/Strongly) ou C (Ensemble), este modelo
utiliza um Qubit Ancilla para controlar a aplicação de camadas de emaranhamento,
criando superposições de caminhos de computação (Interferência Construtiva/Destrutiva).

Arquitetura:
1. Input: Amplitude Embedding (32 features).
2. Controle: Hadamard no Ancilla cria superposição |0> + |1>.
3. Caminho |0>: BasicEntanglerLayers (Pesos A).
4. Operador de Mistura: CNOTs entre Ancilla e Dados.
5. Caminho |1>: BasicEntanglerLayers (Pesos B).
6. Interferência Final: Hadamard no Ancilla.
7. Saída: Probabilidade do Qubit Ancilla.

Objetivo:
Verificar se a capacidade de interferência explícita supera redes neurais quânticas padrão
em dados genômicos complexos.
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
from pennylane import numpy as pnp
from pennylane.templates import AmplitudeEmbedding, BasicEntanglerLayers
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
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

# --- CLASSE DE MONITORAMENTO (Padronizada) ---
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
print(">>> INICIANDO TESTE G: QNN ICO (INTERFERENCE CONTROL OPTIMIZATION)")
print(">>> Config: Ancilla Control | Dual BasicEntangler Path | Hadamard Interference")
print("█"*80)

# --- HIPERPARÂMETROS ---
# ICO usa BasicEntangler que é mais leve, permitindo (teoricamente) mais camadas
# Mas mantemos 2 blocos de profundidade para comparabilidade
CAMADAS_POR_BLOCO = 2   
EPOCHS = 20
BATCH_SIZE = 16
STEPSIZE = 0.02
WIRES_DATA = range(5)   # 32 Features (0,1,2,3,4)
WIRE_ANCILLA = 5        # Qubit de Controle (5)
N_WIRES_TOTAL = 6       # Total

dev = qml.device("default.qubit", wires=N_WIRES_TOTAL)

@qml.qnode(dev)
def qnode_ico(inputs, weights):
    """
    Circuito ICO (Interference Controlled Operation).
    weights shape esperado: (2, n_layers, n_data_wires)
    """
    # 1. Embedding nos dados
    AmplitudeEmbedding(features=inputs, wires=WIRES_DATA, pad_with=0.0, normalize=True)
    
    # 2. Preparação do Controle (Hadamard no Ancilla)
    qml.Hadamard(wires=WIRE_ANCILLA)
    
    # 3. Caminho 0 (Simulado pela aplicação sequencial e controle quântico implícito)
    # Na prática variacional, aplicamos o bloco A
    # BasicEntanglerLayers preserva fase, crucial para interferência
    BasicEntanglerLayers(weights[0], wires=WIRES_DATA, rotation=qml.RY)
    
    # 4. Operador de Mistura (Entanglement entre Controle e Dados)
    # Isso garante que o estado dos dados dependa do ancilla
    for i in WIRES_DATA:
        qml.CNOT(wires=[WIRE_ANCILLA, i])
    
    # 5. Caminho 1 (Bloco B)
    BasicEntanglerLayers(weights[1], wires=WIRES_DATA, rotation=qml.RY)
    
    # 6. Interferência Final (Hadamard fecha o interferômetro)
    qml.Hadamard(wires=WIRE_ANCILLA)
    
    # Medimos a probabilidade do Ancilla estar em |0> ou |1>
    # A classe será determinada por esta probabilidade
    return qml.probs(wires=WIRE_ANCILLA)

def cost_fn(weights, x_batch, y_batch):
    preds = [qnode_ico(x, weights) for x in x_batch]
    loss = 0
    eps = 1e-7
    for p, y in zip(preds, y_batch):
        # Probabilidade da classe 1
        pv = pnp.clip(p[1], eps, 1-eps)
        loss -= (y * pnp.log(pv) + (1-y)*pnp.log(1-pv))
    return loss / len(x_batch)

def predict(weights, X):
    probs = np.array([qnode_ico(x, weights) for x in X])
    return np.where(probs[:, 1] > 0.5, 1, 0)

def balancear_treino(X_train, y_train):
    X_0, X_1 = X_train[y_train == 0], X_train[y_train == 1]
    if len(X_1) == 0: return X_train, y_train
    X_1_up = resample(X_1, replace=True, n_samples=len(X_0), random_state=SEED_GLOBAL)
    X_bal = np.vstack((X_0, X_1_up))
    y_bal = np.hstack((np.zeros(len(X_0)), np.ones(len(X_0))))
    perm = np.random.permutation(len(X_bal))
    return X_bal[perm], y_bal[perm].astype(int)

def executar_teste():
    monitor = ResourceMonitor("Teste_G_QNN_ICO")
    monitor.start()
    
    # Carregar Dados
    X, y = carregar_dados_benchmark(tipo_scaling='quantum')
    
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED_GLOBAL)
    accs = []
    
    # Históricos para gráficos
    history_loss = []
    last_cm = None
    
    print(f"\nIniciando Cross-Validation ({K_FOLDS} Folds) para Modelo G...")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr_orig, y_tr_orig = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # Balanceamento
        X_train, y_train = balancear_treino(X_tr_orig, y_tr_orig)
        
        # Inicialização "Cold Start" (Anti-BP) específica para ICO
        # Shape: (2 caminhos, n_layers, n_wires_data)
        shape = (2, CAMADAS_POR_BLOCO, len(WIRES_DATA))
        # Inicialização pequena gaussiana
        weights = pnp.random.normal(0, 0.01, size=shape, requires_grad=True)
        
        opt = qml.AdamOptimizer(STEPSIZE)
        
        fold_losses = []
        
        # Loop de Treino
        print(f"\n--- Fold {fold+1} ---")
        for epoch in range(EPOCHS):
            perm = np.random.permutation(len(X_train))
            X_train_s, y_train_s = X_train[perm], y_train[perm]
            
            epoch_loss = 0
            steps = 0
            
            for i in range(0, len(X_train), BATCH_SIZE):
                bx = X_train_s[i:i+BATCH_SIZE]
                by = y_train_s[i:i+BATCH_SIZE]
                weights, val = opt.step_and_cost(lambda w: cost_fn(w, bx, by), weights)
                epoch_loss += val
                steps += 1
            
            avg_loss = epoch_loss / steps if steps > 0 else 0
            fold_losses.append(avg_loss)
            
            # --- MONITORAMENTO DETALHADO POR ÉPOCA (VALIDAÇÃO) ---
            # Realiza inferência no conjunto de Teste/Validação deste Fold
            pred_val = predict(weights, X_test)
            
            # Cálculo de Métricas
            cm_val = confusion_matrix(y_test, pred_val, labels=[0, 1])
            tn, fp, fn, tp = cm_val.ravel()
            
            sensibilidade = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Recall
            especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            acuracia_bal = balanced_accuracy_score(y_test, pred_val)
            
            # Log formatado como solicitado
            print(f"   [Ep {epoch+1:02d}] Loss: {avg_loss:.4f} | Acc: {acuracia_bal:.1%} | Sens (Recall): {sensibilidade:.1%} | Espec: {especificidade:.1%}")
        
        history_loss = fold_losses # Guarda o histórico do último fold para plotagem
        
        # Avaliação Final do Fold
        pred_test = predict(weights, X_test)
        score = balanced_accuracy_score(y_test, pred_test)
        accs.append(score)
        last_cm = confusion_matrix(y_test, pred_test)
        
        print(f"   -> Resultado Final Fold {fold+1}: {score:.2%}")
        
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    
    print("\n" + "-"*40)
    print(f"[RESULTADO FINAL MODELO G - ICO]")
    print(f"Média Acurácia Balanceada: {mean_acc:.2%} (+/- {std_acc:.2%})")
    print("-" * 40)
    
    monitor.stop()
    
    # Exportar CSV
    df = pd.DataFrame([{
        'Modelo': 'QNN_ICO_Interference', 
        'Acuracia_Media': mean_acc, 
        'Std_Dev': std_acc,
        'Camadas_Por_Bloco': CAMADAS_POR_BLOCO,
        'Tipo_Ansatz': 'BasicEntangler_DualPath'
    }])
    df.to_csv(os.path.join(PASTA_RESULTADOS, 'resultado_modelo_G.csv'), index=False)
    
    # --- GERAÇÃO DE GRÁFICOS ---
    try:
        print("\n[GRÁFICOS] Gerando curvas do Modelo G...")
        
        # 1. Loss
        plt.figure(figsize=(10, 5))
        plt.plot(history_loss, label='Treino (Loss)', color='teal', linewidth=2)
        plt.title(f'Convergência Modelo G (ICO)\nInterference Controlled Opt')
        plt.xlabel('Época')
        plt.ylabel('Custo')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(PASTA_GRAFICOS, 'modelo_G_convergencia.png'))
        plt.close()
        
        # 2. Confusão
        if last_cm is not None:
            plt.figure(figsize=(6, 5))
            disp = ConfusionMatrixDisplay(confusion_matrix=last_cm, display_labels=['Saudável', 'Doença'])
            # CORREÇÃO: 'Teal' não é um mapa de cores válido no matplotlib. Alterado para 'GnBu' (Green-Blue) que é similar.
            disp.plot(cmap='GnBu', values_format='d')
            plt.title('Matriz de Confusão (Modelo G)')
            plt.savefig(os.path.join(PASTA_GRAFICOS, 'modelo_G_matriz_confusao.png'))
            plt.close()
            
        print(f"[SUCESSO] Gráficos salvos em {PASTA_GRAFICOS}")
        
    except Exception as e:
        print(f"[AVISO] Erro ao gerar gráficos: {e}")

if __name__ == "__main__":
    executar_teste()