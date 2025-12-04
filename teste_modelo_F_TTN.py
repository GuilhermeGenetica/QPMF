# --- teste_modelo_F_TTN.py ---
# -*- coding: utf-8 -*-
"""
MODELO F: Hierarchical Quantum Networks (TTN & MERA)
---------------------------------------------------
Arquivo de Benchmark Avançado - Framework QPMF (2025).
Explora arquiteturas baseadas em Tensor Networks Hierárquicas (Tree Tensor Networks).

Objetivo:
Superar a limitação de localidade do MPS (Modelo D) e a "força bruta" do Modelo A,
usando uma estrutura que mimetiza a hierarquia biológica (Genes -> Vias -> Fenótipo).

Espaço de Teste (Grid Search Automático):
Este script executa 8 pipelines sequenciais para encontrar a configuração campeã:
1. Arquiteturas:
   - TTN (Tree Tensor Network): Coarse-graining direto ($N \to 1$).
   - MERA (Multi-scale Entanglement Renormalization): Adiciona camadas de "Disentanglers" 
     para capturar correlações de longo alcance e reduzir ruído antes da redução.
2. Encodings:
   - Amplitude Embedding (32 Features em 8 Qubits - Alta Densidade).
   - Angle Encoding (PCA 8 Features em 8 Qubits - Estrutura Geométrica).
3. Blocos Variacionais (Ansatz):
   - StronglyEntanglingLayers (Alta Expressividade).
   - BasicEntanglerLayers (Eficiência/Menor Ruído).

Configuração de Hardware:
- 8 Qubits (Potência de 2 ideal para árvores binárias: 8->4->2->1).
"""

import sys
import os
import time
import gc
import psutil
import datetime
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.templates import AngleEmbedding, AmplitudeEmbedding, StronglyEntanglingLayers, BasicEntanglerLayers
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import resample

# Tenta importar codecarbon
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
        print(f"[{datetime.datetime.now()}] Iniciando: {self.task_name}")
        if self.tracker: self.tracker.start()
        
    def stop(self):
        duration = time.time() - self.start_time
        emissions = 0.0
        if self.tracker:
            try: emissions = self.tracker.stop()
            except: pass
        print(f"   [FIM] Duração: {duration:.2f}s | Emissões: {emissions:.6f} kgCO2eq")
        return duration, emissions

print("\n" + "█"*80)
print(">>> INICIANDO TESTE F: HIERARCHICAL TENSOR NETWORKS (TTN & MERA)")
print(">>> Grid Search: 2 Archs x 2 Encodings x 2 Layers = 8 Configurações")
print(">>> MODO ISOLADO: Reset de Seeds e Device a cada iteração.")
print("█"*80)

# --- CONFIGURAÇÃO DO DISPOSITIVO ---
N_QUBITS_TREE = 8
# NOTA: O dispositivo global foi removido. Agora é instanciado localmente por teste.

# ==============================================================================
# 1. BLOCOS CONSTRUTIVOS (GATES)
# ==============================================================================

def Block_Strongly(wires, weights):
    """Bloco denso e expressivo."""
    StronglyEntanglingLayers(weights, wires=wires)

def Block_Basic(wires, weights):
    """Bloco leve e eficiente."""
    BasicEntanglerLayers(weights, wires=wires, rotation=qml.RY)

def get_block_shape(layer_type, n_wires, n_layers_per_block=1):
    """Retorna o shape correto dos pesos para cada tipo de bloco."""
    if layer_type == "Strongly":
        return StronglyEntanglingLayers.shape(n_layers=n_layers_per_block, n_wires=n_wires)
    else:
        # BasicEntangler shape é (n_layers, n_wires)
        return (n_layers_per_block, n_wires)

# ==============================================================================
# 2. ARQUITETURAS HIERÁRQUICAS (TTN & MERA)
# ==============================================================================

def Ansatz_TTN(weights, layer_type):
    """
    Tree Tensor Network (Árvore Binária Invertida).
    Fluxo: 8 -> 4 -> 2 -> 1.
    """
    # Nível 1: 8 Qubits -> 4 Pares ((0,1), (2,3), (4,5), (6,7))
    # Aplicamos bloco unitário em cada par.
    # O qubit da esquerda (par) carrega a informação para cima, o da direita é descartado.
    
    # Pesos divididos por nível. Assumimos que 'weights' é uma lista de tensores.
    
    # --- NÍVEL 1 (8 wires) ---
    if layer_type == "Strongly": Block_Strongly([0, 1], weights[0]); Block_Strongly([2, 3], weights[1]); Block_Strongly([4, 5], weights[2]); Block_Strongly([6, 7], weights[3])
    else: Block_Basic([0, 1], weights[0]); Block_Basic([2, 3], weights[1]); Block_Basic([4, 5], weights[2]); Block_Basic([6, 7], weights[3])
    
    # --- NÍVEL 2 (4 wires ativos: 0, 2, 4, 6) ---
    if layer_type == "Strongly": Block_Strongly([0, 2], weights[4]); Block_Strongly([4, 6], weights[5])
    else: Block_Basic([0, 2], weights[4]); Block_Basic([4, 6], weights[5])
    
    # --- NÍVEL 3 (2 wires ativos: 0, 4) -> Raiz no 0 ---
    if layer_type == "Strongly": Block_Strongly([0, 4], weights[6])
    else: Block_Basic([0, 4], weights[6])

def Ansatz_MERA(weights, layer_type):
    """
    MERA: Adiciona 'Disentanglers' antes de cada redução da TTN.
    Disentanglers operam nas fronteiras dos blocos para remover emaranhamento de curto alcance.
    """
    # Indices dos pesos ajustados para incluir disentanglers
    w_idx = 0
    
    # --- NÍVEL 1 ---
    # Disentanglers (Fronteiras: 1-2, 3-4, 5-6, 7-0(pbc? não, vamos manter open boundary))
    # Aplicamos em (1,2), (3,4), (5,6)
    if layer_type == "Strongly": 
        Block_Strongly([1, 2], weights[w_idx]); w_idx+=1
        Block_Strongly([3, 4], weights[w_idx]); w_idx+=1
        Block_Strongly([5, 6], weights[w_idx]); w_idx+=1
    else:
        Block_Basic([1, 2], weights[w_idx]); w_idx+=1
        Block_Basic([3, 4], weights[w_idx]); w_idx+=1
        Block_Basic([5, 6], weights[w_idx]); w_idx+=1
        
    # Isometries (TTN Layer 1: 0-1, 2-3, 4-5, 6-7)
    if layer_type == "Strongly":
        Block_Strongly([0, 1], weights[w_idx]); w_idx+=1
        Block_Strongly([2, 3], weights[w_idx]); w_idx+=1
        Block_Strongly([4, 5], weights[w_idx]); w_idx+=1
        Block_Strongly([6, 7], weights[w_idx]); w_idx+=1
    else:
        Block_Basic([0, 1], weights[w_idx]); w_idx+=1
        Block_Basic([2, 3], weights[w_idx]); w_idx+=1
        Block_Basic([4, 5], weights[w_idx]); w_idx+=1
        Block_Basic([6, 7], weights[w_idx]); w_idx+=1
        
    # --- NÍVEL 2 (Ativos: 0, 2, 4, 6) ---
    # Disentangler (Fronteira: 2-4)
    if layer_type == "Strongly": Block_Strongly([2, 4], weights[w_idx]); w_idx+=1
    else: Block_Basic([2, 4], weights[w_idx]); w_idx+=1
    
    # Isometries (TTN Layer 2: 0-2, 4-6)
    if layer_type == "Strongly":
        Block_Strongly([0, 2], weights[w_idx]); w_idx+=1
        Block_Strongly([4, 6], weights[w_idx]); w_idx+=1
    else:
        Block_Basic([0, 2], weights[w_idx]); w_idx+=1
        Block_Basic([4, 6], weights[w_idx]); w_idx+=1
        
    # --- NÍVEL 3 (Ativos: 0, 4) ---
    # Isometry Final (0-4) - Sem disentangler necessário aqui (só resta um par)
    if layer_type == "Strongly": Block_Strongly([0, 4], weights[w_idx])
    else: Block_Basic([0, 4], weights[w_idx])

# ==============================================================================
# 3. CONSTRUTOR DE CIRCUITOS (Dynamic QNodes)
# ==============================================================================

def get_qnode(architecture, embedding, layer_type, device_instance):
    """
    Cria um QNode isolado associado a um dispositivo específico.
    """
    @qml.qnode(device_instance)
    def circuit(inputs, weights_flat):
        # 1. Embedding
        if embedding == "Amplitude":
            # Amplitude em 8 qubits = 256 amplitudes. 
            AmplitudeEmbedding(features=inputs, wires=range(N_QUBITS_TREE), pad_with=0.0, normalize=True)
        else: # Angle
            # Angle em 8 qubits requer vetor de tamanho 8.
            AngleEmbedding(features=inputs, wires=range(N_QUBITS_TREE), rotation='Y')
            
        # 2. Reconstrutuir lista de pesos a partir do flat array
        # TTN: 7 blocos | MERA: 11 blocos
        num_blocks = 7 if architecture == "TTN" else 11
        
        single_block_shape = get_block_shape(layer_type, n_wires=2) # Blocos são sempre de 2 qubits
        block_size = np.prod(single_block_shape)
        
        weights_list = []
        for i in range(num_blocks):
            start = i * block_size
            end = start + block_size
            w_block = weights_flat[start:end].reshape(single_block_shape)
            weights_list.append(w_block)
            
        # 3. Ansatz
        if architecture == "TTN":
            Ansatz_TTN(weights_list, layer_type)
        else:
            Ansatz_MERA(weights_list, layer_type)
            
        # 4. Medição na Raiz (Qubit 0)
        return qml.probs(wires=0)
        
    return circuit

# ==============================================================================
# 4. PREPARAÇÃO DE DADOS E FUNÇÕES DE TREINO
# ==============================================================================

def get_data_for_encoding(encoding_type):
    X_raw, y = carregar_dados_benchmark(tipo_scaling='classic') # Raw standard scaled
    
    if encoding_type == "Angle":
        # PCA para 8 componentes (1 por qubit)
        print(f"   [DATA] Aplicando PCA (32 -> {N_QUBITS_TREE}) para Angle Encoding...")
        pca = PCA(n_components=N_QUBITS_TREE, random_state=SEED_GLOBAL)
        X_pca = pca.fit_transform(X_raw)
        # Scaling [0, PI]
        X_final = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X_pca)
        var = np.sum(pca.explained_variance_ratio_)
        print(f"   [DATA] Variância Retida: {var:.2%}")
        return X_final, y, "classic" 
        
    else: # Amplitude
        # Amplitude precisa de normalização L2. 
        X_amp, y_amp = carregar_dados_benchmark(tipo_scaling='quantum')
        print(f"   [DATA] Usando 32 Features completas para Amplitude (Padding autom.).")
        return X_amp, y_amp, "quantum"

def weighted_loss(probs, y_true, w0, w1):
    loss = 0.0
    eps = 1e-7
    for p, y in zip(probs, y_true):
        p_c = pnp.clip(p[1], eps, 1-eps)
        w = w1 if y==1 else w0
        loss -= w * (y * pnp.log(p_c) + (1-y)*pnp.log(1-p_c))
    return loss / len(y_true)

def treinar_configuracao(config, X, y):
    # --- ISOLAMENTO DE TESTE ---
    # 1. Garbage Collection para limpar memória de tensores antigos
    gc.collect()
    
    # 2. Reset de Sementes Aleatórias
    # Isso garante que cada teste comece EXATAMENTE do mesmo ponto estocástico,
    # eliminando a interferência "quem correu antes afeta quem corre depois".
    np.random.seed(SEED_GLOBAL)
    pnp.random.seed(SEED_GLOBAL)
    
    # 3. Criação de Dispositivo Local
    # Dispositivo fresco para evitar cache residual do PennyLane
    dev_local = qml.device("default.qubit", wires=N_QUBITS_TREE)
    
    arch = config['arch']
    emb = config['emb']
    layer = config['layer']
    
    name = f"{arch}_{emb}_{layer}"
    print(f"\n--- Treinando: {name} ---")
    
    # Split (seed fixo garante mesmo split)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED_GLOBAL)
    
    # Pesos de Classe
    n_pos = np.sum(y_tr == 1)
    n_neg = np.sum(y_tr == 0)
    w1 = n_neg / n_pos if n_pos > 0 else 1.0
    w1 *= 1.5 
    print(f"   Pesos Loss: Neg=1.0, Pos={w1:.2f}")
    
    # Setup QNode com Device Local
    qnode = get_qnode(arch, emb, layer, dev_local)
    
    # Inicialização de Pesos
    num_blocks = 7 if arch == "TTN" else 11
    shape_block = get_block_shape(layer, 2)
    total_params = num_blocks * np.prod(shape_block)
    
    # --- HIPERPARÂMETROS ---
    current_stepsize = 0.02 # Default
    
    # >>>> TUNING DEDICADO POR ARQUITETURA <<<<
    # Implementação de IF/ELIF explícito para TODOS os modelos para configuração individual
    
    # 1. TTN + Amplitude + Strongly (MANTIDO - Funcional)
    if arch == "TTN" and emb == "Amplitude" and layer == "Strongly":
        print("   [OTIMIZAÇÃO] TTN Amplitude Strongly: Configuração Agressiva (LR 0.2).")
        current_stepsize = 0.2 
        params = pnp.random.normal(0, 1, size=(total_params,), requires_grad=True)

    # 2. TTN + Amplitude + Basic (MANTIDO - Gentle Nudge)
    elif arch == "TTN" and emb == "Amplitude" and layer == "Basic":
        print("   [OTIMIZAÇÃO] TTN Amplitude Basic: 'Gentle Nudge' (0.49-0.89) + LR 0.01.")
        current_stepsize = 0.01 
        params = pnp.random.uniform(low=0.49, high=0.89, size=(total_params,), requires_grad=True)

    # 3. MERA + Amplitude + Strongly (NOVO - Mais Profundo = Mais Energia Inicial)
    elif arch == "MERA" and emb == "Amplitude" and layer == "Strongly":
        print("   [OTIMIZAÇÃO] MERA Amplitude Strongly: Profundidade Extra -> Init Normal 1.0 + LR 0.2.")
        current_stepsize = 0.2
        # MERA tem mais camadas, precisa de mais variância para não morrer no inicio
        params = pnp.random.normal(0, 1, size=(total_params,), requires_grad=True)

    # 4. MERA + Amplitude + Basic (NOVO - Mais Profundo = Mais Cuidado com Explosão)
    elif arch == "MERA" and emb == "Amplitude" and layer == "Basic":
        print("   [OTIMIZAÇÃO] MERA Amplitude Basic: Estabilidade (LR 0.02) + Gentle Nudge (0.5-1.0).")
        current_stepsize = 0.015 # LR Muito baixo para evitar spikes de 43.0 de Loss
        # Gentle Nudge levemente mais conservador
        params = pnp.random.uniform(low=0.8, high=0.9, size=(total_params,), requires_grad=True)

    # 5. TTN + Angle + Strongly (ADICIONADO - Explícito)
    elif arch == "TTN" and emb == "Angle" and layer == "Strongly":
        print("   [OTIMIZAÇÃO] TTN Angle Strongly: Configuração Padrão/Conservadora.")
        current_stepsize = 0.02
        params = pnp.random.normal(0, 0.05, size=(total_params,), requires_grad=True)

    # 6. TTN + Angle + Basic (ADICIONADO - Explícito)
    elif arch == "TTN" and emb == "Angle" and layer == "Basic":
        print("   [OTIMIZAÇÃO] TTN Angle Basic: Configuração Padrão/Conservadora.")
        current_stepsize = 0.02
        params = pnp.random.normal(0, 0.5, size=(total_params,), requires_grad=True)

    # 7. MERA + Angle + Strongly (ADICIONADO - Explícito)
    elif arch == "MERA" and emb == "Angle" and layer == "Strongly":
        print("   [OTIMIZAÇÃO] MERA Angle Strongly: Configuração Padrão/Conservadora.")
        current_stepsize = 0.02
        params = pnp.random.normal(0, 0.05, size=(total_params,), requires_grad=True)

    # 8. MERA + Angle + Basic (ADICIONADO - Explícito)
    elif arch == "MERA" and emb == "Angle" and layer == "Basic":
        print("   [OTIMIZAÇÃO] MERA Angle Basic: Configuração Padrão/Conservadora.")
        current_stepsize = 0.07
        params = pnp.random.normal(0.6, 0.8, size=(total_params,), requires_grad=True)

    # Fallback (Segurança)
    else:
        print("   [OTIMIZAÇÃO] Default Fallback.")
        current_stepsize = 0.02
        params = pnp.random.normal(0, 0.05, size=(total_params,), requires_grad=True)

    opt = qml.AdamOptimizer(current_stepsize)
    
    last_loss = 0
    # Treino
    for ep in range(20):
        idx = np.random.choice(len(X_tr), 20)
        x_b, y_b = X_tr[idx], y_tr[idx]
        
        params, last_loss = opt.step_and_cost(lambda w: weighted_loss([qnode(x, w) for x in x_b], y_b, 1.0, w1), params)
        
        # Monitoramento
        probs_ep = np.array([qnode(x, params)[1] for x in X_te])
        preds_ep = np.where(probs_ep > 0.5, 1, 0)
        acc_ep = balanced_accuracy_score(y_te, preds_ep)
        
        n_pos_te = np.sum(y_te == 1)
        if n_pos_te > 0:
            recall_ep = np.sum((preds_ep == 1) & (y_te == 1)) / n_pos_te
        else:
            recall_ep = 0.0
            
        print(f"   Ep {ep+1}: Loss {last_loss:.4f} | Acc: {acc_ep:.2%} | Recall: {recall_ep:.2%}")
        
    print("") # Newline
    
    # Avaliação
    probs = np.array([qnode(x, params)[1] for x in X_te])
    preds = np.where(probs > 0.5, 1, 0)
    
    acc = balanced_accuracy_score(y_te, preds)
    report = classification_report(y_te, preds, output_dict=True, zero_division=0)
    recall = report['1']['recall']
    
    print(f"   RESULTADO: Acc={acc:.2%} | Recall(Doença)={recall:.2%}")
    
    # Limpeza final do teste
    del qnode
    del dev_local
    gc.collect()
    
    return {
        'name': name,
        'acc': acc,
        'recall': recall,
        'params': params,
        'X_te': X_te,
        'y_te': y_te,
        'history_loss': float(last_loss)
    }

# ==============================================================================
# 5. DIAGNÓSTICO CAUSAL (Para o Vencedor)
# ==============================================================================
def diagnostico_final(best_res):
    print(f"\n>>> ANALISANDO VENCEDOR: {best_res['name']}")
    
    # Recria o QNode e Device apenas para o diagnóstico (Isolamento)
    if "TTN" in best_res['name']: arch = "TTN"
    else: arch = "MERA"
    
    if "Amplitude" in best_res['name']: emb = "Amplitude"
    else: emb = "Angle"
    
    if "Strongly" in best_res['name']: layer = "Strongly"
    else: layer = "Basic"
    
    dev_diag = qml.device("default.qubit", wires=N_QUBITS_TREE)
    qnode = get_qnode(arch, emb, layer, dev_diag)
    
    params = best_res['params']
    X_te = best_res['X_te']
    y_te = best_res['y_te']
    
    # Matriz de Confusão
    preds = np.where(np.array([qnode(x, params)[1] for x in X_te]) > 0.5, 1, 0)
    cm = confusion_matrix(y_te, preds)
    
    try:
        plt.figure(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Saudável', 'Doença'])
        disp.plot(cmap='GnBu', values_format='d')
        plt.title(f"Melhor Modelo F: {best_res['name']}")
        plt.savefig(os.path.join(PASTA_GRAFICOS, 'modelo_F_best_confusao.png'))
        plt.close()
        print("   Gráfico salvo.")
    except Exception as e:
        print(f"   [AVISO] Erro ao salvar gráfico: {e}")

    recall = best_res['recall']
    print(f"\n   [MÉTRICA] Recall Final: {recall:.2%}")

    if recall >= 0.95:
        print("   [CLASSIFICAÇÃO] Padrão Ouro (Diagnóstico). Quase inalcançável hoje sem hardware real perfeito.")
    elif recall >= 0.80:
        print("   [CLASSIFICAÇÃO] Excelente / Publicável. Estado da arte em QML. Resultado extraordinário.")
    elif recall >= 0.70:
        print("   [CLASSIFICAÇÃO] Promissor / Satisfatório. Capturou a estrutura dos dados.")
    elif recall >= 0.50:
        print("   [CLASSIFICAÇÃO] Insuficiente para Diagnóstico. O modelo luta para generalizar. Requer ajustes.")
    else:
        print("   [CLASSIFICAÇÃO] Falha Crítica. O modelo está pior que jogar uma moeda.")

# ==============================================================================
# 6. MAIN LOOP
# ==============================================================================

if __name__ == "__main__":
    monitor = ResourceMonitor("Benchmark_Modelo_F")
    monitor.start()
    
    # Preparar Dados
    X_angle, y_angle, _ = get_data_for_encoding("Angle")
    X_amp, y_amp, _ = get_data_for_encoding("Amplitude")
    
    # Grid de Configurações
    archs = ["TTN", "MERA"] 
    embs = ["Amplitude", "Angle"]
    layers = ["Strongly", "Basic"]
    
    results = []
    
    for arch, emb, layer in itertools.product(archs, embs, layers):
        config = {'arch': arch, 'emb': emb, 'layer': layer}
        
        # Selecionar dados corretos
        X_curr = X_amp if emb == "Amplitude" else X_angle
        y_curr = y_amp 
        
        try:
            res = treinar_configuracao(config, X_curr, y_curr)
            results.append(res)
        except Exception as e:
            print(f"   [ERRO] Falha no treino de {arch}_{emb}_{layer}: {e}")
        
    # Leaderboard
    print("\n" + "="*60)
    print("LEADERBOARD MODELO F (TTN/MERA)")
    print("="*60)
    
    if len(results) > 0:
        results.sort(key=lambda x: (x['recall'], x['acc']), reverse=True)
        
        df_res = pd.DataFrame([{k: v for k, v in r.items() if k not in ['params', 'X_te', 'y_te']} for r in results])
        df_res = df_res.rename(columns={'name': 'Modelo', 'acc': 'Acuracia_Media', 'recall': 'Recall'})
        
        print(df_res[['Modelo', 'Acuracia_Media', 'Recall']].to_string(index=False))
        df_res.to_csv(os.path.join(PASTA_RESULTADOS, 'resultado_modelo_F_grid.csv'), index=False)
        
        diagnostico_final(results[0])

        try:
            print("\n[GRÁFICOS] Gerando análise visual comparativa (Grid Search)...")
            plt.figure(figsize=(12, 6))
            names = df_res['Modelo']
            accs = df_res['Acuracia_Media']
            recalls = df_res['Recall']
            x = np.arange(len(names))
            width = 0.35
            rects1 = plt.bar(x - width/2, accs, width, label='Acurácia', color='skyblue')
            rects2 = plt.bar(x + width/2, recalls, width, label='Recall (Doença)', color='salmon')
            plt.xlabel('Configuração (Arch_Encoding_Ansatz)')
            plt.ylabel('Score')
            plt.title('Performance Comparativa: TTN vs MERA (Modelo F)')
            plt.xticks(x, names, rotation=45, ha='right')
            plt.ylim(0, 1.05)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            plt.tight_layout()
            grafico_path = os.path.join(PASTA_GRAFICOS, 'modelo_F_comparacao_grid.png')
            plt.savefig(grafico_path)
            plt.close()
            print(f"   [SUCESSO] Gráfico comparativo salvo em: {grafico_path}")
        except Exception as e:
            print(f"   [AVISO] Erro ao gerar gráfico comparativo: {e}")
    else:
        print("Nenhum modelo completou o treino com sucesso.")
    
    monitor.stop()