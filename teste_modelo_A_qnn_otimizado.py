# --- teste_modelo_A_qnn_hibrido.py ---
# -*- coding: utf-8 -*-
"""
MODELO A: QNN Híbrido (Best of Both Worlds)
-----------------------------------------------------------------
Arquivo de Teste Unitário - Integrante do Framework QPMF.
Fusão de  Resultados Altos com a engenharia (Logs/Scheduler).

BASEADO EM: teste_modelo_A_qnn_otimizado anterior.py (Preservando integridade)
MELHORIAS: Monitoramento detalhado e Scheduler de refinamento.

AJUSTES DE RECUPERAÇÃO (Back to Basics):
- Learning Rate: Retornado para 0.01 (Estabilidade comprovada).
- Regularização: Removida a L2 (Permitir plasticidade total aos pesos).
- Scheduler: Ajustado para Refinamento Fino (Fine-tuning), não correção.
- Épocas: 40 (Equilíbrio entre convergência e tempo).

Configuração Técnica:
- Embedding: Amplitude Embedding (32 features -> 5 qubits).
- Ansatz: StronglyEntanglingLayers (6 qubits para emaranhamento denso).
- Estratégia Anti-BP: Inicialização "Cold Start" (Gaussiana Estreita).
- Otimizador: Adam com Taxa de Aprendizado Adaptativa (Scheduler).
- Validação: Stratified K-Fold com Oversampling para classes desbalanceadas.
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
from pennylane.templates import AmplitudeEmbedding, StronglyEntanglingLayers
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

# --- CLASSE DE MONITORAMENTO (Detalhada) ---
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
print(">>> INICIANDO TESTE A: QNN HÍBRIDO")
print(">>> Config: LR 0.02 (Estável) | Sem L2 | Scheduler de Refino | 3 Camadas")
print("█"*80)

# --- HIPERPARÂMETROS HÍBRIDOS ---
CAMADAS = 3             # Mantido 3 para expressividade (Improvement)
EPOCHS = 40             # Ajustado: Mais que 30, menos que 75 ( +-40)
BATCH_SIZE = 16         # Batch pequeno para generalização estocástica
STEPSIZE_INIT = 0.02    # RESTAURADO: 0.02 provou ser o valor ideal nos testes anteriores
WIRES_DATA = range(5)   # 2^5 = 32 features (Qubits 0 a 4)
WIRES_TOTAL = range(N_WIRES) # Qubits 0 a 5 (6 Qubits total - Config Comum)

dev = qml.device("default.qubit", wires=N_WIRES)

@qml.qnode(dev)
def qnode_model_a(inputs, weights):
    """
    Circuito Variacional Híbrido.
    1. AmplitudeEmbedding: Comprime 32 floats em 5 qubits (Logaritmicamente eficiente).
    2. StronglyEntanglingLayers: Ansatz heurístico denso para capturar correlações complexas.
       Aplica-se a TODOS os 6 qubits, espalhando a informação dos dados para o qubit ancilla.
    """
    # Carregamento de Dados (Estado |psi>)
    # pad_with=0.0 garante que se vetor < 32, preenche. normalize=True garante norma L2=1.
    AmplitudeEmbedding(features=inputs, wires=WIRES_DATA, pad_with=0.0, normalize=True)
    
    # Processamento Variacional (Camadas de Rotação + CNOTs all-to-all)
    StronglyEntanglingLayers(weights, wires=WIRES_TOTAL)
    
    # Medição do Qubit 0 (Pode ser qualquer um, 0 é convenção)
    return qml.probs(wires=0)

def cost_fn(weights, x_batch, y_batch):
    """
    Função de Custo Log-Loss (Cross Entropy) para Classificação Binária.
    NOTA: Regularização L2 removida nesta versão Híbrida para permitir máxima plasticidade.
    """
    loss = 0
    # Inferência em Batch (Pode ser paralelizada em hardware real)
    preds = [qnode_model_a(x, weights) for x in x_batch]
    
    for p, y in zip(preds, y_batch):
        # Probabilidade da classe 1 (Positiva/Doença)
        p_val = p[1] 
        # Clipping para evitar log(0)
        p_val = pnp.clip(p_val, 1e-7, 1 - 1e-7)
        
        # Binary Cross Entropy
        loss -= (y * pnp.log(p_val) + (1 - y) * pnp.log(1 - p_val))
        
    return loss / len(x_batch)

def predict(weights, X):
    """Predição de classes (Hard Voting 0.5 threshold)."""
    probs = np.array([qnode_model_a(x, weights) for x in X])
    # probs[:, 1] é a probabilidade da classe 1
    return np.where(probs[:, 1] > 0.5, 1, 0)

def balancear_treino(X_train, y_train):
    """
    Estratégia de Oversampling para mitigar desbalanceamento de classes (VUS Raras).
    Apenas no TREINO, nunca no teste.
    """
    X_0, X_1 = X_train[y_train == 0], X_train[y_train == 1]
    
    # Proteção contra folds sem classe positiva (raro mas possível)
    if len(X_1) == 0: 
        print("   [AVISO] Fold sem classe positiva no treino. Mantendo original.")
        return X_train, y_train
        
    # Upsample da classe minoritária (1) para igualar a majoritária (0)
    X_1_up = resample(X_1, replace=True, n_samples=len(X_0), random_state=SEED_GLOBAL)
    
    X_bal = np.vstack((X_0, X_1_up))
    y_bal = np.hstack((np.zeros(len(X_0)), np.ones(len(X_0))))
    
    # Embaralhar
    perm = np.random.permutation(len(X_bal))
    return X_bal[perm], y_bal[perm].astype(int)

def executar_teste():
    monitor = ResourceMonitor("Teste_A_QNN_Hibrido")
    monitor.start()
    
    # Carga de Dados (Scaling Quantum = Normalização L2 para Amplitude Embedding)
    X, y = carregar_dados_benchmark(tipo_scaling='quantum')
    
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED_GLOBAL)
    accs = []
    
    # Para plotagem (usaremos o último fold como representativo)
    history_loss = []
    last_cm = None
    
    print(f"\nIniciando Cross-Validation ({K_FOLDS} Folds)...")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n>>> Fold {fold+1}/{K_FOLDS}")
        
        X_tr_orig, y_tr_orig = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # Balanceamento Crítico
        X_train, y_train = balancear_treino(X_tr_orig, y_tr_orig)
        print(f"   Treino Balanceado: {len(X_train)} amostras (Neg: {np.sum(y_train==0)}, Pos: {np.sum(y_train==1)})")
        
        # Inicialização "Cold Start" (Anti-Barren Plateau)
        # Inicializa pesos com desvio padrão muito baixo (0.01) para começar próximo da identidade
        # Isso facilita o fluxo de gradientes nas primeiras épocas.
        shape = StronglyEntanglingLayers.shape(n_layers=CAMADAS, n_wires=N_WIRES)
        weights = pnp.random.normal(loc=0.0, scale=0.01, size=shape, requires_grad=True)
        
        # Otimizador com Scheduler Híbrido (Começa Robusto 0.01)
        current_lr = STEPSIZE_INIT
        opt = qml.AdamOptimizer(stepsize=current_lr)
        
        best_loss_fold = float('inf')
        fold_losses = []
        
        t_fold = time.time()
        for epoch in range(EPOCHS):
            
            # --- LR SCHEDULER (Melhoria aplicada ao Híbrido) ---
            # Ajuste Fino: Só reduz após a metade do treino para garantir exploração inicial
            if epoch == 20: 
                current_lr = STEPSIZE_INIT * 0.5 # Cai para 0.005
                opt = qml.AdamOptimizer(stepsize=current_lr)
                # print(f"   [Scheduler] Refinamento: LR reduzido para {current_lr:.4f}")
            elif epoch == 35:
                # Estágio final de refinamento (opcional, mas bom para garantir convergência)
                current_lr = STEPSIZE_INIT * 0.2 
                opt = qml.AdamOptimizer(stepsize=current_lr)

            # Shuffle por época
            perm = np.random.permutation(len(X_train))
            X_train_s, y_train_s = X_train[perm], y_train[perm]
            
            loss_ep = 0
            steps = 0
            
            # Loop de Mini-batch
            for i in range(0, len(X_train), BATCH_SIZE):
                bx = X_train_s[i:i+BATCH_SIZE]
                by = y_train_s[i:i+BATCH_SIZE]
                
                weights, val = opt.step_and_cost(lambda w: cost_fn(w, bx, by), weights)
                loss_ep += val
                steps += 1
            
            avg_loss = loss_ep / steps
            fold_losses.append(avg_loss)
            if avg_loss < best_loss_fold: best_loss_fold = avg_loss
            
            # --- MONITORAMENTO (Métricas Detalhadas durante o treino) ---
            # A cada 5 épocas fazemos check completo (mais frequente que o original 10)
            if (epoch+1) % 5 == 0 or epoch == 0:
                pred_val = predict(weights, X_test)
                
                # Cálculo detalhado de Sensibilidade e Especificidade
                cm_val = confusion_matrix(y_test, pred_val, labels=[0, 1])
                tn, fp, fn, tp = cm_val.ravel()
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                acc = balanced_accuracy_score(y_test, pred_val)
                
                print(f"    [Ep {epoch+1:02d}] Loss: {avg_loss:.4f} | Acc: {acc:.1%} | Sens: {sens:.1%} | Espec: {spec:.1%}")

        # Guardar histórico do último fold
        history_loss = fold_losses

        # Avaliação Final do Fold
        pred_test = predict(weights, X_test)
        score = balanced_accuracy_score(y_test, pred_test)
        accs.append(score)
        
        cm = confusion_matrix(y_test, pred_test)
        last_cm = cm
        print(f"   Resultado Final Fold {fold+1}: Bal. Acc = {score:.2%} ({time.time()-t_fold:.1f}s)")
        print(f"   Matriz Confusão: {cm.tolist()}")

    # Estatísticas Finais
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    
    print("\n" + "-"*40)
    print(f"[RESULTADO FINAL MODELO A - HÍBRIDO]")
    print(f"Média Acurácia Balanceada: {mean_acc:.2%} (+/- {std_acc:.2%})")
    print("-" * 40)
    
    monitor.stop()
    
    # Exportação para o Benchmark Master
    df = pd.DataFrame([{
        'Modelo': 'QNN_Otimizado_Hibrido', 
        'Acuracia_Media': mean_acc, 
        'Std_Dev': std_acc,
        'Camadas': CAMADAS,
        'Features': N_FEATURES
    }])
    df.to_csv(os.path.join(PASTA_RESULTADOS, 'resultado_modelo_A_hibrido.csv'), index=False)

    # --- GERAÇÃO DE GRÁFICOS ---
    try:
        print("\n[GRÁFICOS] Gerando curvas de convergência do Modelo A Híbrido...")
        
        # 1. Curva de Loss
        plt.figure(figsize=(10, 5))
        plt.plot(history_loss, label='Treino (Loss Original)', color='green', linewidth=2)
        plt.title(f'Convergência Modelo A (Híbrido - LR 0.01)\nÚltimo Fold - {EPOCHS} Épocas')
        plt.xlabel('Época')
        plt.ylabel('Função de Custo')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(PASTA_GRAFICOS, 'modelo_A_hibrido_convergencia.png'))
        plt.close()
        
        # 2. Matriz de Confusão
        if last_cm is not None:
            plt.figure(figsize=(6, 5))
            disp = ConfusionMatrixDisplay(confusion_matrix=last_cm, display_labels=['Saudável', 'Doença'])
            disp.plot(cmap='Greens', values_format='d')
            plt.title('Matriz de Confusão Híbrida (Último Fold)')
            plt.savefig(os.path.join(PASTA_GRAFICOS, 'modelo_A_hibrido_cm.png'))
            plt.close()
            
        print(f"[SUCESSO] Gráficos salvos em {PASTA_GRAFICOS}")
    except Exception as e:
        print(f"[AVISO] Erro ao gerar gráficos: {e}")

if __name__ == "__main__":
    executar_teste()