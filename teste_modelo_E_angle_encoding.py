# --- teste_modelo_E_angle_encoding.py ---
# -*- coding: utf-8 -*-
"""
MODELO E: Angle Encoding com Redução Dimensional Otimizada (V3 - 8 Qubits Stable)
---------------------------------------------------------------------------------
Arquivo de Teste Unitário - Integrante do Framework QPMF.
Versão Estabilizada para 8 Qubits (2025).

ATUALIZAÇÃO V3 (Estabilidade 8Q):
- Diagnóstico: A versão 8Q mostrou alto potencial (pico 91%) mas alta instabilidade 
  (média 69%) devido à dificuldade de otimização no espaço de Hilbert maior.
- Solução: 
  1. Aumento de Épocas (20 -> 50) para garantir convergência.
  2. LR Scheduler (Decaimento) para refinar a busca e escapar de mínimos locais.
  3. Manutenção dos 8 Qubits para capturar 35%+ da variância biológica.

ATUALIZAÇÃO V4 (Monitoramento Granular):
- Ajuste: Learning Rate inicial mais agressivo (0.05) para escapar de mínimos locais.
- Ajuste: Scheduler de 3 estágios.
- Monitoramento: Logs detalhados (Acc, Sens, Spec, Cost) por época.

Contexto Teórico:
Mapeamos as 8 principais componentes principais (PCA) em rotações de 8 Qubits.
Isso cria um compromisso ideal entre expressividade e treinabilidade.
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
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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

# --- CONFIGURAÇÃO LOCAL ---
# 8 Qubits para capturar ~35% da variância (Sweet Spot)
N_WIRES_E = 8 

print("\n" + "█"*80)
print(f">>> INICIANDO TESTE E: ANGLE ENCODING (V4 - 8Q MONITORADO)")
print(f">>> Config: {N_WIRES_E} Qubits | 50 Épocas | LR Scheduler Agressivo")
print("█"*80)

# --- HIPERPARÂMETROS V4 ---
CAMADAS = 3             
EPOCHS = 50             # Mantido alto para permitir convergência no espaço maior
BATCH_SIZE = 16         
STEPSIZE_INIT = 0.05    # LR Inicial mais agressivo (Era 0.02) para romper barreiras iniciais

dev = qml.device("default.qubit", wires=N_WIRES_E)

@qml.qnode(dev)
def qnode_model_e(inputs, weights):
    """
    Circuito Angle Encoding.
    inputs: Vetor de 8 elementos (PCA reduzido).
    """
    # Angle Embedding: Codifica características em rotações Y
    AngleEmbedding(features=inputs, wires=range(N_WIRES_E), rotation='Y')
    
    # Processamento Variacional
    StronglyEntanglingLayers(weights, wires=range(N_WIRES_E))
    
    return qml.probs(wires=0)

def cost_fn(weights, x_batch, y_batch):
    loss = 0
    preds = [qnode_model_e(x, weights) for x in x_batch]
    
    for p, y in zip(preds, y_batch):
        p_val = pnp.clip(p[1], 1e-7, 1 - 1e-7)
        loss -= (y * pnp.log(p_val) + (1 - y) * pnp.log(1 - p_val))
        
    return loss / len(x_batch)

def predict(weights, X):
    probs = np.array([qnode_model_e(x, weights) for x in X])
    return np.where(probs[:, 1] > 0.5, 1, 0)

def preprocess_angle_encoding(X_raw):
    """
    Pipeline específico para Angle Encoding:
    1. PCA: Reduz 32 -> 8 dimensões.
    2. MinMaxScaler: Escala para [0, π].
    """
    print(f"   [PRE-PROCESSAMENTO] Reduzindo dimensão {X_raw.shape[1]} -> {N_WIRES_E} via PCA...")
    
    from sklearn.preprocessing import StandardScaler
    scaler_std = StandardScaler()
    X_std = scaler_std.fit_transform(X_raw)
    
    pca = PCA(n_components=N_WIRES_E, random_state=SEED_GLOBAL)
    X_pca = pca.fit_transform(X_std)
    
    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"   [PCA] Variância Explicada com {N_WIRES_E} componentes: {explained_var:.2%}")
    
    scaler_angle = MinMaxScaler(feature_range=(0, np.pi))
    X_final = scaler_angle.fit_transform(X_pca)
    
    return X_final

def balancear_treino(X_train, y_train):
    X_0, X_1 = X_train[y_train == 0], X_train[y_train == 1]
    if len(X_1) == 0: return X_train, y_train
    X_1_up = resample(X_1, replace=True, n_samples=len(X_0), random_state=SEED_GLOBAL)
    X_bal = np.vstack((X_0, X_1_up))
    y_bal = np.hstack((np.zeros(len(X_0)), np.ones(len(X_0))))
    perm = np.random.permutation(len(X_bal))
    return X_bal[perm], y_bal[perm].astype(int)

def executar_teste():
    monitor = ResourceMonitor("Teste_E_AngleEncoding_8Q_V4")
    monitor.start()
    
    X_raw, y = carregar_dados_benchmark(tipo_scaling='classic')
    X = preprocess_angle_encoding(X_raw)
    
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED_GLOBAL)
    accs = []
    
    history_loss = []
    last_cm = None
    
    print(f"\nIniciando Cross-Validation ({K_FOLDS} Folds)...")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr_orig, y_tr_orig = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        X_train, y_train = balancear_treino(X_tr_orig, y_tr_orig)
        
        # Inicialização
        shape = StronglyEntanglingLayers.shape(n_layers=CAMADAS, n_wires=N_WIRES_E)
        weights = pnp.random.normal(loc=0.0, scale=0.01, size=shape, requires_grad=True)
        
        # Otimizador com Scheduler Manual
        opt = qml.AdamOptimizer(stepsize=STEPSIZE_INIT)
        
        fold_losses = []
        t_fold = time.time()
        
        print(f"\n>>> Fold {fold+1}")
        
        for epoch in range(EPOCHS):
            # --- LR Scheduler Gradual ---
            # Começa com 0.05. Reduz progressivamente para refinar.
            if epoch == 15: opt = qml.AdamOptimizer(stepsize=0.02)
            if epoch == 30: opt = qml.AdamOptimizer(stepsize=0.01)
            if epoch == 40: opt = qml.AdamOptimizer(stepsize=0.005)

            perm = np.random.permutation(len(X_train))
            X_train_s, y_train_s = X_train[perm], y_train[perm]
            
            epoch_loss = 0
            steps = 0
            
            for i in range(0, len(X_train), BATCH_SIZE):
                bx = X_train_s[i:i+BATCH_SIZE]
                by = y_train_s[i:i+BATCH_SIZE]
                weights, loss_val = opt.step_and_cost(lambda w: cost_fn(w, bx, by), weights)
                epoch_loss += loss_val
                steps += 1
            
            avg_loss = epoch_loss / steps if steps > 0 else 0
            fold_losses.append(avg_loss)
            
            # --- MONITORAMENTO DETALHADO POR ÉPOCA ---
            # Realiza inferência no conjunto de Teste/Validação deste Fold
            pred_val = predict(weights, X_test)
            
            # Cálculo de Métricas Completas
            cm_val = confusion_matrix(y_test, pred_val, labels=[0, 1])
            tn, fp, fn, tp = cm_val.ravel()
            
            sensibilidade = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Recall
            especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            acc_bal = balanced_accuracy_score(y_test, pred_val)
            
            # Log formatado linha a linha
            print(f"   [Ep {epoch+1:02d}] Custo: {avg_loss:.4f} | Acc: {acc_bal:.2%} | Sens: {sensibilidade:.2%} | Espec: {especificidade:.2%}")
        
        history_loss = fold_losses
        
        # Avaliação Final do Fold
        pred_test = predict(weights, X_test)
        score = balanced_accuracy_score(y_test, pred_test)
        accs.append(score)
        last_cm = confusion_matrix(y_test, pred_test)
        
        print(f"   -> Resultado Final Fold {fold+1}: {score:.2%} ({time.time()-t_fold:.1f}s)")

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    
    print("\n" + "-"*40)
    print(f"[RESULTADO FINAL MODELO E - 8 QUBITS V4]")
    print(f"Média Acurácia Balanceada: {mean_acc:.2%} (+/- {std_acc:.2%})")
    print("-" * 40)
    
    monitor.stop()
    
    df = pd.DataFrame([{
        'Modelo': 'Angle_Encoding_PCA_8Q_V4', 
        'Acuracia_Media': mean_acc, 
        'Std_Dev': std_acc,
        'Camadas': CAMADAS,
        'Dim_Reduzida': N_WIRES_E
    }])
    df.to_csv(os.path.join(PASTA_RESULTADOS, 'resultado_modelo_E.csv'), index=False)

    # --- GERAÇÃO DE GRÁFICOS ---
    try:
        print("\n[GRÁFICOS] Gerando curvas de análise do Modelo E...")
        
        if history_loss:
            plt.figure(figsize=(10, 5))
            plt.plot(history_loss, label='Treino (Log Loss)', color='purple', linewidth=2)
            plt.title(f'Convergência Modelo E (8Q V4)\nÚltimo Fold - {EPOCHS} Épocas')
            plt.xlabel('Época')
            plt.ylabel('Função de Custo')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join(PASTA_GRAFICOS, 'modelo_E_convergencia_loss.png'))
            plt.close()
        
        if last_cm is not None:
            plt.figure(figsize=(6, 5))
            disp = ConfusionMatrixDisplay(confusion_matrix=last_cm, display_labels=['Saudável', 'Doença'])
            disp.plot(cmap='Purples', values_format='d')
            plt.title('Matriz de Confusão (Último Fold)')
            plt.savefig(os.path.join(PASTA_GRAFICOS, 'modelo_E_matriz_confusao.png'))
            plt.close()
            
        print(f"[SUCESSO] Gráficos salvos em {PASTA_GRAFICOS}")
    except Exception as e:
        print(f"[AVISO] Erro ao gerar gráficos: {e}")

if __name__ == "__main__":
    executar_teste()