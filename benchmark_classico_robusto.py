# --- benchmark_classico_robusto.py ---
# -*- coding: utf-8 -*-
"""
SUITE DE TESTES CLÁSSICOS (BASELINE) - VERSÃO FINAL 2025
--------------------------------------------------------
Benchmark de referência para comparação com modelos quânticos.
Otimizado para dados genômicos tabulares (N=500, D=32).

Status Atual:
- RandomForest e LogReg lideram (~77%).
- Estrutura dos dados confirmada como predominantemente Linear/Aditiva.

Alterações V3:
- [FIX] KNN agora usa PCA interno para combater a maldição da dimensionalidade.
- [TUNING] GradientBoosting recalibrado para recuperar performance (Depth 2 -> 3).
- [AUDIT] Mantida auditoria de energia e tempo.
- [VISUAL] Geração de gráficos de ranking automático.
"""

import os
import sys
import time
import psutil
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Scikit-Learn Imports
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score

# Tenta importar codecarbon
try:
    from codecarbon import EmissionsTracker
    HAS_CODECARBON = True
except ImportError:
    HAS_CODECARBON = False

# Importação do Config Comum
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config_comum import *

# --- MONITORAMENTO ---
class ResourceMonitor:
    def __init__(self, task_name):
        self.task_name = task_name
        self.tracker = EmissionsTracker(project_name=task_name, log_level='error') if HAS_CODECARBON else None
        self.start_time = None
        
    def start(self):
        self.start_time = time.time()
        print(f"[{datetime.datetime.now()}] Iniciando: {self.task_name}")
        print(f"   Recursos Iniciais: CPU {psutil.cpu_percent()}% | RAM {psutil.virtual_memory().percent}%")
        if self.tracker: self.tracker.start()
        
    def stop(self):
        duration = time.time() - self.start_time
        emissions = 0.0
        if self.tracker:
            try: emissions = self.tracker.stop()
            except: pass
        
        print(f"[{datetime.datetime.now()}] Finalizado: {self.task_name}")
        print(f"   Duração: {duration:.2f}s | Energia Est.: {emissions:.6f} kgCO2eq")
        return duration, emissions

print("\n" + "█"*80)
print(">>> INICIANDO BENCHMARK CLÁSSICO (VERSÃO FINAL)")
print(">>> Foco: Validação Cruzada Robusta e Comparabilidade Quântica")
print("█"*80)

def executar_benchmark_classico():
    # Carregar dados
    X, y = carregar_dados_benchmark(tipo_scaling='classic')
    
    # --- DEFINIÇÃO DOS MODELOS ---
    
    # 1. KNN OTIMIZADO (Pipeline com PCA)
    # Resolve o problema dos 50%: reduz 32 dimensões para 10 antes de calcular distâncias.
    knn_pipeline = Pipeline([
        ('pca', PCA(n_components=10, random_state=SEED_GLOBAL)),
        ('knn', KNeighborsClassifier(n_neighbors=9, weights='distance', metric='cosine', n_jobs=-1))
    ])

    modelos = {
        # CAMPEÃO ATUAL 1
        'LogReg': LogisticRegression(
            max_iter=5000,        
            solver='lbfgs',       
            C=1.0, 
            class_weight='balanced',
            random_state=SEED_GLOBAL
        ),
        
        'KNN_PCA': knn_pipeline,
        
        # CONFIRMADO: Linear funciona melhor que RBF para estes dados
        'SVM_Linear': SVC(
            kernel='linear',      
            C=1.0,               
            probability=True, 
            class_weight='balanced',
            random_state=SEED_GLOBAL
        ),
        
        # CAMPEÃO ATUAL 2 (Limitado para evitar Overfitting)
        'RandomForest': RandomForestClassifier(
            n_estimators=300,
            max_depth=5,                 
            min_samples_leaf=4,          
            class_weight='balanced_subsample',
            n_jobs=-1,
            random_state=SEED_GLOBAL
        ),
        
        # RECUPERAÇÃO: Aumentando capacidade de aprendizado
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,          
            max_depth=3,                 # Retornado para 3 (padrão)
            subsample=0.8,               # Stochastic Gradient Boosting (reduz variância)
            random_state=SEED_GLOBAL
        ),
        
        'MLP_Net': MLPClassifier(
            hidden_layer_sizes=(64, 32, 16), # Arquitetura funil
            learning_rate_init=0.001,
            activation='relu',
            solver='adam',
            alpha=0.01,                  
            max_iter=10000, 
            early_stopping=True,
            validation_fraction=0.1,
            random_state=SEED_GLOBAL
            
        )
    }
    
    # Ensemble: Une os melhores especialistas
    modelos['Voting_Ensemble'] = VotingClassifier(
        estimators=[
            ('rf', modelos['RandomForest']), 
            ('svm', modelos['SVM_Linear']), 
            ('lr', modelos['LogReg'])
        ],
        voting='soft',
        n_jobs=-1
    )

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED_GLOBAL)
    resultados_consolidados = []

    for nome, clf in modelos.items():
        print(f"\n" + "-"*60)
        monitor = ResourceMonitor(f"Modelo_{nome}")
        monitor.start()
        
        scores_fold = []
        
        for i, (train_idx, test_idx) in enumerate(kf.split(X, y)):
            t0 = time.time()
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            
            score = balanced_accuracy_score(y_test, pred)
            scores_fold.append(score)
            
            dt = time.time() - t0
            
            # Log minimalista para não poluir
            print(f"   > Fold {i+1:02d}: Acc = {score:.2%} ({dt:.3f}s)")
            
        duration, emissions = monitor.stop()
        
        mean_score = np.mean(scores_fold)
        std_score = np.std(scores_fold)
        
        print(f"   [RESULTADO {nome}]: Média = {mean_score:.2%} (+/- {std_score:.2%})")
        
        resultados_consolidados.append({
            'Modelo': nome,
            'Acuracia_Media': mean_score,
            'Std_Dev': std_score,
            'Tempo_Total_s': duration,
            'Energia_kgCO2': emissions,
            'Scores_Raw': scores_fold
        })

    print("\n" + "="*80)
    print("RANKING FINAL (CLÁSSICO)")
    print("="*80)
    
    df_pd = pd.DataFrame(resultados_consolidados).sort_values(by='Acuracia_Media', ascending=False)
    melhor = df_pd.iloc[0]
    
    # Tabela formatada
    print(f"{'MODELO':<20} | {'MÉDIA':<10} | {'STD':<10} | {'TEMPO (s)':<10}")
    print("-" * 60)
    for _, row in df_pd.iterrows():
        print(f"{row['Modelo']:<20} | {row['Acuracia_Media']:.2%}   | {row['Std_Dev']:.2%}   | {row['Tempo_Total_s']:.2f}")
    
    print("-" * 60)
    print(f"CAMPEÃO: {melhor['Modelo']} ({melhor['Acuracia_Media']:.2%})")

    # Comparação Estatística
    print(f"\n--- VALIDAÇÃO ESTATÍSTICA (vs {melhor['Modelo']}) ---")
    for idx, row in df_pd.iloc[1:].iterrows():
        t_stat, p_val = stats.ttest_rel(melhor['Scores_Raw'], row['Scores_Raw'])
        sig = "SIM" if p_val < 0.05 else "NÃO"
        print(f"  vs {row['Modelo']:<18}: p={p_val:.4f} | Diferença Significativa? {sig}")

    # Salvar
    caminho_csv = os.path.join(PASTA_RESULTADOS, 'resultados_classicos_completos.csv')
    df_pd.drop(columns=['Scores_Raw']).to_csv(caminho_csv, index=False)
    print(f"\n[SUCESSO] Resultados salvos em: {caminho_csv}")

    # --- GERAÇÃO DE GRÁFICOS (ADICIONADO) ---
    try:
        print("\n[GRÁFICOS] Gerando visualização do Ranking Clássico...")
        plt.figure(figsize=(12, 7))
        
        # Cores baseadas no score
        colors = plt.cm.viridis(df_pd['Acuracia_Media'] / df_pd['Acuracia_Media'].max())
        
        bars = plt.barh(df_pd['Modelo'], df_pd['Acuracia_Media'], xerr=df_pd['Std_Dev'], color=colors, capsize=5, alpha=0.8)
        
        plt.xlabel('Acurácia Balanceada Média')
        plt.title('Benchmark Clássico: Comparação de Performance (10-Fold CV)')
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.xlim(0, 1.05)
        
        # Adicionar valores nas barras
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.1%}', 
                     va='center', ha='left', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        caminho_fig = os.path.join(PASTA_GRAFICOS, 'ranking_classico_benchmark.png')
        plt.savefig(caminho_fig)
        plt.close()
        print(f"[SUCESSO] Gráfico salvo em: {caminho_fig}")
    except Exception as e:
        print(f"[AVISO] Não foi possível gerar gráficos: {e}")

if __name__ == "__main__":
    executar_benchmark_classico()