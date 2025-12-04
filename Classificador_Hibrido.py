# --- Classificador_Hibrido.py ---
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import vcfpy
import random
import sys
import os
import time
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

# --- CONFIGURAÇÕES DE CAMINHOS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NOME_VCF = os.path.join(BASE_DIR, 'dados_geneticos.vcf')
NOME_CSV = os.path.join(BASE_DIR, 'fenotipos.csv')
NOME_MAP_QUANTUM = os.path.join(BASE_DIR, 'mapeamento_variantes_quantum.csv')

ARQUIVO_X = os.path.join(BASE_DIR, 'X_hibrido.npy')
ARQUIVO_Y = os.path.join(BASE_DIR, 'y_hibrido.npy')
ARQUIVO_INDICES = os.path.join(BASE_DIR, 'indices_vus_selecionadas.txt')
ARQUIVO_PRIORS = os.path.join(BASE_DIR, 'priors_vus_selecionadas.npy')
ARQUIVO_RELATORIO = os.path.join(BASE_DIR, 'relatorio_vus_selecionadas.txt')

# --- HIPERPARÂMETROS DE SELEÇÃO ---
N_PRE_POOL = 120          # Aumentado para permitir maior diversidade antes do GA
LD_THRESHOLD = 0.80       # Tolerância a LD levemente maior para não perder drivers
N_FEATURES_FINAL = 32     # Exato para Amplitude Embedding (5 Qubits)

# --- PARÂMETROS DO GA (HQGA) ---
TAMANHO_POPULACAO = 80    # Aumentado para explorar melhor o espaço
NUM_GERACOES = 50
TAXA_CROSSOVER = 0.90
TAXA_MUTACAO = 0.10
ELITISMO = 4              # Preserva os 4 melhores absolutos (Sem regressão)
PENALIDADE_TAMANHO = 0.5  # Penalidade rígida

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print(f"\n" + "█"*80)
print(f">>> CLASSIFICADOR HÍBRIDO (HQGA-VUS) COM SUPORTE HWE & 32-FEATURES")
print(f">>> Target: {N_FEATURES_FINAL} Features | Seed: {SEED} | Pool Inicial: {N_PRE_POOL}")
print("█"*80 + "\n")

# ==============================================================================
# 1. FUNÇÕES AUXILIARES E ESTATÍSTICA GENÔMICA
# ==============================================================================

def calcular_hwe_pvalue(genotypes):
    """
    Calcula o P-Valor do Equilíbrio de Hardy-Weinberg (HWE).
    Entrada: Array de genótipos (0, 1, 2).
    Saída: P-valor (Baixo = Desvio/Erro Técnico ou Seleção Forte).
    """
    obs_aa = np.sum(genotypes == 0)
    obs_ab = np.sum(genotypes == 1)
    obs_bb = np.sum(genotypes == 2)
    n = len(genotypes)
    
    if n == 0: return 1.0
    
    # Frequências alélicas
    p = (2 * obs_aa + obs_ab) / (2 * n)
    q = 1 - p
    
    # Esperados
    exp_aa = (p ** 2) * n
    exp_ab = (2 * p * q) * n
    exp_bb = (q ** 2) * n
    
    # Chi-Square (Evita divisão por zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        chisq, p_val = chisquare([obs_aa, obs_ab, obs_bb], f_exp=[exp_aa, exp_ab, exp_bb])
    
    if np.isnan(p_val): return 1.0
    return p_val

def carregar_mapa_discordancia_32d():
    if os.path.exists(NOME_MAP_QUANTUM):
        try:
            print(f"[IO] Carregando Metadados 32D: {NOME_MAP_QUANTUM}")
            df = pd.read_csv(NOME_MAP_QUANTUM)
            if 'Discordancia' in df.columns:
                max_disc = df['Discordancia'].max()
                if max_disc > 0:
                    df['Score_Norm'] = df['Discordancia'] / max_disc
                else:
                    df['Score_Norm'] = 0.0
                return dict(zip(df['Variant_ID'], df['Score_Norm']))
        except Exception as e:
            print(f"[AVISO] Falha ao ler metadados: {e}")
    return {}

def calcular_priors_avancados(vcf_reader, X_matrix, map_discordancia=None):
    """
    Calcula Scores (Priors) considerando:
    1. Anotação Biológica (VCF).
    2. Entropia das 32 Features (Map Discordância).
    3. Penalidade por violação de Hardy-Weinberg (Controle de Qualidade).
    """
    n_total = X_matrix.shape[1]
    priors = np.zeros(n_total)
    variant_ids_list = []
    
    # Pesos
    W_32D = 8.0
    W_EXONIC = 2.0
    W_CLINVAR = 3.0
    W_CADD = 1.5
    PENALIDADE_HWE = 0.1 # Multiplicador (Reduz score em 90% se falhar no HWE)
    
    print(f"Calculando Priors para {n_total} variantes (HWE + Biologia)...")
    
    start_t = time.time()
    
    for i, record in enumerate(vcf_reader):
        score = 1.0
        
        # ID
        try:
            var_id = f"{record.CHROM}:{record.POS}:{record.REF}:{record.ALT[0].value}"
        except:
            var_id = f"var_{i}"
        variant_ids_list.append(var_id)
        
        # 1. Discordância 32D
        if map_discordancia and var_id in map_discordancia:
            score += (map_discordancia[var_id] * W_32D)
            
        # 2. HWE Check (Quality Control)
        # Se desvio for muito extremo (p < 1e-6), provável erro de sequenciamento
        # Penalizamos para evitar que o modelo aprenda ruído técnico
        p_hwe = calcular_hwe_pvalue(X_matrix[:, i])
        if p_hwe < 1e-6:
            score *= PENALIDADE_HWE
            
        # 3. Anotações Padrão
        info = record.INFO
        func = info.get('Func.refGene', ['.'])[0]
        if func in ['exonic', 'splicing']: score += W_EXONIC
        
        clnsig = info.get('CLNSIG', ['.'])[0]
        if 'Pathogenic' in clnsig: score += W_CLINVAR
        
        try:
            cadd = float(info.get('CADD_phred', [0])[0])
            if cadd > 20: score += W_CADD
        except: pass
        
        priors[i] = score
        
        if i % 1000 == 0 and i > 0:
            print(f"   Processado {i}/{n_total}...", end='\r')
            
    print(f"\nPriors Calculados em {time.time()-start_t:.2f}s.")
    
    scaler = MinMaxScaler(feature_range=(1, 10))
    priors_norm = scaler.fit_transform(priors.reshape(-1, 1)).flatten()
    return priors_norm, variant_ids_list

def carregar_dataset_completo():
    print("[IO] Carregando VCF e CSV...")
    if not os.path.exists(NOME_VCF) or not os.path.exists(NOME_CSV):
        raise FileNotFoundError("Inputs não encontrados.")
    
    df_pheno = pd.read_csv(NOME_CSV)
    y = df_pheno['Status_Doenca'].values
    nomes_amostras = df_pheno['ID_Amostra'].values
    
    reader = vcfpy.Reader.from_path(NOME_VCF)
    records = list(reader)
    n_vars = len(records)
    n_samples = len(y)
    
    X = np.zeros((n_samples, n_vars), dtype=np.int8)
    sample_map = {name: i for i, name in enumerate(nomes_amostras)}
    
    for j, rec in enumerate(records):
        for call in rec.calls:
            s_idx = sample_map.get(call.sample)
            if s_idx is not None:
                gt = call.data.get('GT')
                val = 0
                if gt in ['0/1', '0|1', '1/0']: val = 1
                elif gt in ['1/1', '1|1']: val = 2
                X[s_idx, j] = val
                
    return X, y, records

# ==============================================================================
# 2. KERNEL TARGET ALIGNMENT (FITNESS FUNCTION)
# ==============================================================================

class KernelTargetAlignment:
    def __init__(self, X_data, y_labels):
        self.X = X_data
        # Matriz de Alvo Ideal (yyT)
        # y deve ser centralizado ou {-1, 1} para melhor performance do KTA
        y_centered = y_labels - np.mean(y_labels)
        self.Ky = np.outer(y_centered, y_centered)
        self.denom_ky = np.linalg.norm(self.Ky, 'fro')

    def compute_kta(self, feature_indices):
        if len(feature_indices) == 0: return 0.0
        
        X_subset = self.X[:, feature_indices]
        
        # Kernel Linear (Proxy Eficiente)
        # K = XX^T
        K = np.dot(X_subset, X_subset.T)
        
        # KTA Score
        numerator = np.sum(K * self.Ky) # Trace(K * Ky)
        denom_k = np.linalg.norm(K, 'fro')
        
        if denom_k == 0 or self.denom_ky == 0: return 0.0
        
        return numerator / (denom_k * self.denom_ky)

# ==============================================================================
# 3. ALGORITMO GENÉTICO HÍBRIDO (HQGA)
# ==============================================================================

def executar_hqga(X_pool, y, priors_pool, indices_originais_pool):
    n_features_pool = X_pool.shape[1]
    kta_engine = KernelTargetAlignment(X_pool, y)
    
    # Probabilidades de seleção inicial baseadas nos Priors (Bias Biológico)
    probs_prior = priors_pool / np.sum(priors_pool)
    
    # Inicialização
    populacao = []
    for _ in range(TAMANHO_POPULACAO):
        # Amostragem sem reposição ponderada
        ind = np.random.choice(n_features_pool, N_FEATURES_FINAL, replace=False, p=probs_prior)
        populacao.append(list(ind))
        
    best_sol = None
    best_fit = -np.inf
    history = []
    
    print(f"\nIniciando Evolução GA ({NUM_GERACOES} gerações)...")
    
    start_ga = time.time()
    for gen in range(NUM_GERACOES):
        scores = []
        for chrom in populacao:
            fit = kta_engine.compute_kta(chrom)
            # Penalidade suave se tamanho divergir (segurança)
            if len(chrom) != N_FEATURES_FINAL: fit *= 0.5
            scores.append(fit)
            
            if fit > best_fit:
                best_fit = fit
                best_sol = chrom
        
        history.append(best_fit)
        
        # Log Periódico
        if (gen+1) % 10 == 0:
            print(f"   Gen {gen+1:02d}: Melhor KTA = {best_fit:.5f} | Tempo: {time.time()-start_ga:.1f}s")
            
        # Elitismo
        sorted_pop = [x for _, x in sorted(zip(scores, populacao), key=lambda pair: pair[0], reverse=True)]
        nova_pop = sorted_pop[:ELITISMO]
        
        # Reprodução
        while len(nova_pop) < TAMANHO_POPULACAO:
            # Torneio
            p1 = sorted_pop[np.random.randint(0, len(sorted_pop)//2)] # Bias para top 50%
            p2 = sorted_pop[np.random.randint(0, len(sorted_pop))]
            
            # Crossover Uniforme
            child = []
            pool_genes = list(set(p1 + p2))
            # Tenta manter genes com maior prioridade biológica dentro dos pais
            random.shuffle(pool_genes)
            child = pool_genes[:N_FEATURES_FINAL]
            
            # Mutação Guiada (Reintroduz genes do pool global com base no Prior)
            if random.random() < TAXA_MUTACAO:
                idx_mut = random.randint(0, N_FEATURES_FINAL-1)
                # Tenta inserir um gene novo de alto valor biológico
                novo_gene = np.random.choice(n_features_pool, p=probs_prior)
                if novo_gene not in child:
                    child[idx_mut] = novo_gene
            
            nova_pop.append(child)
            
        populacao = nova_pop
        
    indices_globais = [indices_originais_pool[i] for i in best_sol]
    return indices_globais, best_fit, history

# ==============================================================================
# 4. PIPELINE MESTRE
# ==============================================================================

def executar_pipeline_hibrido():
    t0 = time.time()
    
    # 1. Carregar Metadados 32D
    map_32d = carregar_mapa_discordancia_32d()
    
    # 2. Carregar Dados Brutos
    X_full, y, records = carregar_dataset_completo()
    
    # 3. Calcular Priors com HWE
    priors, _ = calcular_priors_avancados(records, X_full, map_32d)
    
    # 4. Pré-Filtro (Funil): Mutual Information + Prior
    print(f"\n[FUNIL] Selecionando Top {N_PRE_POOL} candidatos (MI + Prior)...")
    mi = mutual_info_classif(X_full, y, random_state=SEED)
    # Normaliza MI
    mi_norm = MinMaxScaler().fit_transform(mi.reshape(-1,1)).flatten()
    # Normaliza Priors
    prior_norm = MinMaxScaler().fit_transform(priors.reshape(-1,1)).flatten()
    
    # Score Combinado (60% Biologia/HWE, 40% Estatística Pura)
    combined = 0.6 * prior_norm + 0.4 * mi_norm
    
    indices_sorted = np.argsort(combined)[::-1]
    indices_pre = indices_sorted[:N_PRE_POOL]
    
    X_pool = X_full[:, indices_pre]
    priors_pool = priors[indices_pre]
    
    # 5. LD Pruning (Remover redundância linear)
    print(f"[LD] Removendo variantes com correlação > {LD_THRESHOLD}...")
    df_pool = pd.DataFrame(X_pool)
    corr = df_pool.corr().abs()
    drop_mask = np.zeros(len(indices_pre), dtype=bool)
    
    for i in range(len(indices_pre)):
        if drop_mask[i]: continue
        for j in range(i+1, len(indices_pre)):
            if drop_mask[j]: continue
            if corr.iloc[i, j] > LD_THRESHOLD:
                # Remove a que tiver menor prior biológico
                if priors_pool[i] < priors_pool[j]:
                    drop_mask[i] = True
                else:
                    drop_mask[j] = True
                    
    indices_ld = [i for i in range(len(indices_pre)) if not drop_mask[i]]
    # Se pruning for agressivo demais, recupera os melhores para ter mínimo para o GA
    if len(indices_ld) < N_FEATURES_FINAL:
        print("[AVISO] LD Pruning excessivo. Recuperando features para completar população.")
        indices_ld = list(range(min(len(indices_pre), N_FEATURES_FINAL + 10)))
        
    indices_pool_final = indices_pre[indices_ld]
    X_pool_final = X_full[:, indices_pool_final]
    priors_pool_final = priors[indices_pool_final]
    
    print(f"   Pool Pós-LD: {len(indices_pool_final)} variantes.")
    
    # 6. GA
    idx_final, score_kta, hist = executar_hqga(X_pool_final, y, priors_pool_final, indices_pool_final)
    
    print("\n" + "="*60)
    print(f"SELEÇÃO CONCLUÍDA EM {time.time()-t0:.1f}s")
    print(f"Score KTA Final: {score_kta:.5f}")
    print(f"Features Selecionadas: {len(idx_final)}")
    print("="*60)
    
    # Salvar
    X_final = X_full[:, idx_final]
    priors_final = priors[idx_final]
    
    np.save(ARQUIVO_X, X_final)
    np.save(ARQUIVO_Y, y)
    np.save(ARQUIVO_PRIORS, priors_final)
    np.savetxt(ARQUIVO_INDICES, idx_final, fmt='%d')
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(hist, marker='o', color='purple')
    plt.title(f"Convergência HQGA (Seed {SEED})")
    plt.xlabel("Geração")
    plt.ylabel("KTA")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(BASE_DIR, 'hqga_optimization_log.png'))
    
    # Relatório Detalhado
    with open(ARQUIVO_RELATORIO, 'w') as f:
        f.write("=== RELATÓRIO DE SELEÇÃO HQGA (2025) ===\n")
        f.write(f"KTA Final: {score_kta:.5f}\n")
        f.write(f"HWE Filter: Ativo (Penalidade < 1e-6)\n\n")
        f.write("Variantes Selecionadas:\n")
        for i, real_idx in enumerate(idx_final):
            try:
                vid = records[real_idx].ID[0]
            except: vid = f"idx_{real_idx}"
            p_val = priors[real_idx]
            f.write(f"{i+1:02d}. {vid:<15} | Prior Score: {p_val:.2f}\n")
            
    print(f"[OUTPUT] Arquivos gerados em {BASE_DIR}")

if __name__ == "__main__":
    executar_pipeline_hibrido()