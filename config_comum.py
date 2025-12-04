# --- config_comum.py ---
# -*- coding: utf-8 -*-
"""
CONFIGURAÇÃO COMUM (GLOBAL) PARA O FRAMEWORK QPMF
-------------------------------------------------
Arquivo mestre de configuração.
CORREÇÃO: Tratamento robusto de NaNs e Zeros para evitar crash no SVM/AmplitudeEmbedding.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

# --- 1. DEFINIÇÃO DE CAMINHOS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Inputs (Conexão direta com a saída do Classificador_Hibrido.py)
ARQUIVO_X = os.path.join(BASE_DIR, 'X_hibrido.npy')
ARQUIVO_Y = os.path.join(BASE_DIR, 'y_hibrido.npy')
ARQUIVO_INDICES = os.path.join(BASE_DIR, 'indices_vus_selecionadas.txt')
ARQUIVO_PRIORS = os.path.join(BASE_DIR, 'priors_vus_selecionadas.npy')

# Cache (Otimização para o Modelo B - Dual Kernel)
ARQUIVO_CACHE_KERNEL = os.path.join(BASE_DIR, 'cache_kernel_global_amp.npy')

# Outputs (Centralização para o BENCHMARK_MASTER.py)
PASTA_RESULTADOS = os.path.join(BASE_DIR, 'RESULTADOS_QPMF')
PASTA_GRAFICOS = os.path.join(PASTA_RESULTADOS, 'Graficos')

os.makedirs(PASTA_RESULTADOS, exist_ok=True)
os.makedirs(PASTA_GRAFICOS, exist_ok=True)

# --- 2. CONSTANTES GLOBAIS ---
SEED_GLOBAL = 42

# Amplitude Embedding: 32 Features -> 5 Qubits (+1 Ancilla = 6)
N_FEATURES = 32
N_WIRES = 6 

# Parâmetros de Treino
TEST_SIZE = 0.2
K_FOLDS = 5

# --- 3. CARREGAMENTO DE DADOS ---
def carregar_dados_benchmark(tipo_scaling='quantum'):
    """
    Carrega X e y com tratamento de erros numéricos (NaN/Inf).
    Centraliza a lógica de carregamento para garantir que todos os modelos
    (A, B, C, D e Clássico) vejam EXATAMENTE os mesmos dados.
    """
    if not os.path.exists(ARQUIVO_X) or not os.path.exists(ARQUIVO_Y):
        raise FileNotFoundError(f"Ficheiros de dados não encontrados em {BASE_DIR}. Execute 'Classificador_Hibrido.py' primeiro.")

    # Carrega como float32 para eficiência e compatibilidade com Pennylane
    X = np.load(ARQUIVO_X).astype(np.float32)
    y = np.load(ARQUIVO_Y).astype(np.int64)
    
    # 1. Limpeza Prévia (NaNs -> 0)
    # Proteção contra erros de cálculo no pré-processamento upstream
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 2. Verificação de Dimensão e Padding (Robustez contra Algoritmo Genético variável)
    if X.shape[1] != N_FEATURES:
        print(f"[AVISO] Ajustando dimensão: Input {X.shape[1]} -> Target {N_FEATURES}")
        if X.shape[1] < N_FEATURES:
            # Padding com zeros se faltarem features
            padding = np.zeros((X.shape[0], N_FEATURES - X.shape[1]))
            X = np.hstack((X, padding))
        else:
            # Truncagem se houver excesso
            X = X[:, :N_FEATURES] 

    # 3. Scaling Seguro
    if tipo_scaling == 'quantum':
        # Amplitude Embedding exige norma L2 = 1.
        # Se um vetor for tudo zero, Normalizer falha ou gera NaN. Adicionamos epsilon.
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        # Onde a norma é zero, substituímos por 1 (evita divisão por zero, vetor continua zero)
        norms[norms == 0] = 1.0
        X = X / norms
        
    elif tipo_scaling == 'classic':
        # Padronização Z-Score para algoritmos clássicos (SVM, MLP)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
    # Limpeza Pós-Scaling (Garantia final para o hardware quântico)
    X = np.nan_to_num(X, nan=0.0)

    return X, y

print(f"[CONFIG] Ambiente QPMF pronto. Base: {BASE_DIR}")