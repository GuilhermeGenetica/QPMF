# --- teste_modelo_C_boosting.py ---
# -*- coding: utf-8 -*-
"""
MODELO C: Quantum Boosting (AdaBoost-QNN)
-----------------------------------------
Arquivo de Teste Unitário - Integrante do Framework QPMF.
Atualizado com Estabilidade Numérica SAMME e Auditoria de Recursos (2025).

Configuração Técnica:
- Arquitetura: Ensemble Sequencial de QNNs "Fracos" (Weak Learners).
- Weak Learner: Amplitude Embedding + 1 Camada BasicEntangler (Raso, Rápido e Preserva Fase).
- Estratégia de Boosting: AdaBoost com Reamostragem Ponderada (Weighted Resampling).
- Estabilidade: Suavização de Alpha para evitar explosão de gradientes em erros próximos de 0.
- Objetivo: Transformar classificadores quânticos apenas "pouco melhores que o acaso" em um classificador forte.
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
from pennylane.templates import AmplitudeEmbedding, StronglyEntanglingLayers, BasicEntanglerLayers
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

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
print(">>> INICIANDO TESTE C: QUANTUM BOOSTING (ADABOOST-QNN)")
print(">>> Config: 7 Weak Learners | Resampling Ponderado | SAMME.R Logic")
print(">>> Ansatz: BasicEntanglerLayers (Otimizado para Evitar Barren Plateaus em Ensembles)")
print("█"*80)

# Configurações
N_ESTIMATORS = 15        # Aumentado para 15 para maior robustez do ensemble
LAYERS_WEAK = 2         # Apenas 2 camadas para garantir que seja um "Weak" learner rápido
WIRES_DATA = range(5)   # 32 features
dev = qml.device("default.qubit", wires=N_WIRES)

@qml.qnode(dev)
def weak_qnn(inputs, weights):
    """
    Circuito 'Weak Learner'.
    Deve ser leve e rápido. Amplitude Embedding + 1 Camada de Emaranhamento Básico.
    MUDANÇA ESTRATÉGICA: BasicEntanglerLayers usado ao invés de Strongly.
    Motivo: Preservar gradientes e evitar overfitting no learner fraco.
    """
    AmplitudeEmbedding(features=inputs, wires=WIRES_DATA, pad_with=0.0, normalize=True)
    # Usa BasicEntanglerLayers para manter a simplicidade e fluxo de gradiente
    # CORREÇÃO CRÍTICA: rotation deve ser a classe qml.RY (não string)
    BasicEntanglerLayers(weights, wires=range(N_WIRES), rotation=qml.RY)
    return qml.probs(wires=0)

def cost_fn_weak(weights, x_b, y_b):
    """Cost function padrão para o weak learner (Binary Cross Entropy)."""
    # y_b aqui deve ser 0 ou 1
    preds = [weak_qnn(x, weights) for x in x_b]
    loss = 0
    epsilon = 1e-7
    for p, y in zip(preds, y_b):
        pv = pnp.clip(p[1], epsilon, 1 - epsilon)
        loss -= (y * pnp.log(pv) + (1-y)*pnp.log(1-pv))
    return loss / len(x_b)

def executar_teste():
    monitor = ResourceMonitor("Teste_C_Boosting")
    monitor.start()
    
    # Carregar Dados
    X, y_raw = carregar_dados_benchmark(tipo_scaling='quantum')
    
    # AdaBoost Clássico trabalha com labels {-1, 1} para o cálculo de Alpha
    # Mas o QNN (Binary Cross Entropy) trabalha melhor com {0, 1}
    # Manteremos dois vetores de y sincronizados.
    y_ada = np.where(y_raw == 0, -1, 1) # Para o Boosting update
    y_qnn = y_raw                       # Para o treino do circuito
    
    # Split simples 70/30 (Boosting é custoso para Cross-Validation completa neste benchmark rápido)
    X_train, X_test, y_train_ada, y_test_ada = train_test_split(
        X, y_ada, test_size=0.3, random_state=SEED_GLOBAL, stratify=y_ada
    )
    # Versão 0/1 para treino
    y_train_qnn = np.where(y_train_ada == -1, 0, 1)
    
    n_samples = len(X_train)
    # Inicializa pesos das amostras uniformemente: w_i = 1/N
    sample_weights = np.full(n_samples, 1.0 / n_samples)
    
    estimators = []
    alphas = []
    
    print(f"\nTreinando Ensemble de {N_ESTIMATORS} Estimadores Quânticos (Basic)...")
    
    for m in range(N_ESTIMATORS):
        t_start_est = time.time()
        
        # 1. Resampling Ponderado (O coração do AdaBoost em ML Quântico)
        indices_boot = np.random.choice(
            np.arange(n_samples), size=n_samples, replace=True, p=sample_weights
        )
        X_boot = X_train[indices_boot]
        y_boot_qnn = y_train_qnn[indices_boot]
        
        # 2. Treinar Weak Learner
        # Inicialização "Cold Start" (Anti-BP)
        # ATENÇÃO: Shape do Basic é (n_layers, n_wires), diferente do Strongly (n_layers, n_wires, 3)
        shape = BasicEntanglerLayers.shape(n_layers=LAYERS_WEAK, n_wires=N_WIRES)
        w_m = pnp.random.normal(0, 0.01, size=shape, requires_grad=True)
        opt = qml.AdamOptimizer(stepsize=0.04) # Learning rate agressivo para convergência rápida
        
        # Treino Curto (É um weak learner, não precisa ser perfeito)
        BATCH_BOOST = 32
        for ep in range(8): # 8 Épocas apenas
            # Batch único estocástico para velocidade
            idx_b = np.random.choice(len(X_boot), BATCH_BOOST)
            x_b = X_boot[idx_b]
            y_b = y_boot_qnn[idx_b]
            w_m, loss = opt.step_and_cost(lambda v: cost_fn_weak(v, x_b, y_b), w_m)
            
        # 3. Avaliar no Dataset de Treino Original (Para calcular erro epsilon)
        # Inferência em bloco para velocidade (simulada)
        preds_prob = np.array([weak_qnn(x, w_m)[1] for x in X_train])
        # Predição do estimador m (-1 ou 1)
        preds_ada = np.where(preds_prob > 0.5, 1, -1)
        
        # 4. Calcular Erro Ponderado (Epsilon)
        # Soma dos pesos das amostras onde o modelo errou
        is_error = (preds_ada != y_train_ada)
        err_m = np.sum(sample_weights[is_error])
        
        # Proteção Numérica SAMME (evita log(0) ou divisão por zero)
        err_m = np.clip(err_m, 1e-10, 1 - 1e-10)
        
        # 5. Calcular Alpha (Peso do Estimador)
        # Alpha = 0.5 * ln((1-err)/err)
        alpha_m = 0.5 * np.log((1 - err_m) / err_m)
        
        # 6. Atualizar Pesos das Amostras
        # w_{new} = w_{old} * exp(-alpha * y_true * y_pred)
        update_factor = np.exp(-alpha_m * y_train_ada * preds_ada)
        sample_weights *= update_factor
        
        # Renormalizar pesos para somar 1
        sample_weights /= np.sum(sample_weights)
        
        estimators.append(w_m)
        alphas.append(alpha_m)
        
        # Log
        acc_train_weak = np.mean(preds_ada == y_train_ada)
        print(f"   [Estimador {m+1}] Tempo: {time.time()-t_start_est:.1f}s | Erro Pond: {err_m:.4f} | Alpha: {alpha_m:.4f} | Acc Bruta: {acc_train_weak:.1%}")

    # --- AVALIAÇÃO FINAL (ENSEMBLE) ---
    print("\nCalculando Voto Majoritário Ponderado no Teste...")
    final_preds_score = np.zeros(len(X_test))
    
    for m in range(N_ESTIMATORS):
        # Probabilidade do modelo m
        probs_test = np.array([weak_qnn(x, estimators[m])[1] for x in X_test])
        # Voto (-1 ou 1)
        vote_test = np.where(probs_test > 0.5, 1, -1)
        # Acumula voto ponderado
        final_preds_score += alphas[m] * vote_test
        
    # Sinal final do ensemble
    final_class_ada = np.sign(final_preds_score)
    
    # Métricas
    # Converter para 0/1 para usar sklearn balanced_accuracy
    y_test_bin = np.where(y_test_ada == -1, 0, 1)
    y_pred_bin = np.where(final_class_ada == -1, 0, 1)
    
    acc = balanced_accuracy_score(y_test_bin, y_pred_bin)
    cm = confusion_matrix(y_test_bin, y_pred_bin)
    
    print("\n" + "-"*40)
    print(f"[RESULTADO FINAL MODELO C]")
    print(f"Acurácia Balanceada (Ensemble): {acc:.2%}")
    print(f"Matriz de Confusão:\n{cm}")
    print("-" * 40)
    
    monitor.stop()
    
    # Salvar
    df = pd.DataFrame([{
        'Modelo': 'Quantum_Boosting_Ada', 
        'Acuracia_Media': acc, 
        'Std_Dev': 0.0, # Teste único (hold-out)
        'N_Estimators': N_ESTIMATORS,
        'Estimator_Depht': LAYERS_WEAK
    }])
    df.to_csv(os.path.join(PASTA_RESULTADOS, 'resultado_modelo_C.csv'), index=False)

    # --- GERAÇÃO DE GRÁFICOS (ADICIONADO) ---
    try:
        print("\n[GRÁFICOS] Gerando análise visual do Boosting...")
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, N_ESTIMATORS+1), alphas, color='orange', alpha=0.7)
        plt.title('Importância dos Estimadores (Alphas)\nQuanto maior, mais confiável é o classificador')
        plt.xlabel('Estimador Sequencial')
        plt.ylabel('Peso Alpha (SAMME)')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PASTA_GRAFICOS, 'modelo_C_boosting_weights.png'))
        plt.close()
        print(f"[SUCESSO] Gráfico salvo em {PASTA_GRAFICOS}")
    except Exception as e:
        print(f"[AVISO] Erro ao gerar gráficos: {e}")

if __name__ == "__main__":
    executar_teste()