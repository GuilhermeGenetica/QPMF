# -*- coding: utf-8 -*-
"""
PRE-PROCESSADOR GENÔMICO PARA QUANTUM MACHINE LEARNING (32 FEATURES)
--------------------------------------------------------------------
Objetivo: Preparar variantes VUS do VCF para Amplitude Embedding.
Arquitetura: Funil de Priorização Híbrido (Deduplicação -> Filtro Clássico -> Seleção Entrópica).

Entrada: 'dados_geneticos.vcf' (Gerado pelo simulador ou anotado via VEP/dbNSFP)
Saída: 'X_quantum_32dim.npy' (Matriz de amplitudes normalizadas para o circuito quântico)
       'mapeamento_variantes.csv' (Para retornar resultados aos pacientes)

Features: 32 variáveis de alta densidade (ACMG/AMP Guidelines).
Autor: Pipeline Quântico Otimizado
Data: 2025
"""

import vcfpy
import numpy as np
import pandas as pd
import os
import sys
import warnings

# --- CONFIGURAÇÕES DE AMBIENTE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOME_VCF_ENTRADA = os.path.join(BASE_DIR, 'dados_geneticos.vcf')
NOME_X_QUANTUM = os.path.join(BASE_DIR, 'X_quantum_32dim.npy')
NOME_MAP_SAIDA = os.path.join(BASE_DIR, 'mapeamento_variantes_quantum.csv')

# Filtros do "Firewall Clássico"
LIMITE_AF_BA1 = 0.01  # 1% (GnomAD) - Acima disso é Benigno (BA1)
LIMITE_DISCORDANCIA = 0.15  # Desvio padrão mínimo para considerar "difícil/ambíguo"
TOP_N_SELECAO = 1000  # Máximo de variantes para enviar ao Quantum (Cota de Qubits)

# Semente para reprodutibilidade em imputações
np.random.seed(42)

print("--- INICIANDO PRÉ-PROCESSAMENTO QUÂNTICO (32 DIMENSÕES) ---")

def parse_float(val, default=0.0):
    """Converte valores do VCF (strings/listas) para float de forma segura."""
    try:
        if isinstance(val, list):
            val = val[0]
        if val == '.' or val is None:
            return default
        return float(val)
    except:
        return default

def parse_str(val, default=""):
    """Extrai string limpa de listas/campos do VCF."""
    if isinstance(val, list):
        val = val[0]
    if val == '.' or val is None:
        return default
    return str(val)

# --- FUNÇÕES DE TRANSFORMAÇÃO MATEMÁTICA (FEATURE ENGINEERING) ---

def trans_log_invert(x):
    """Para Frequências: Valor alto (raro) -> Score alto. Baixo (comum) -> Score baixo."""
    # x é frequencia (0 a 1). Log10 de x pequeno é negativo grande.
    # -log10(1e-6) = 6. -log10(0.1) = 1.
    epsilon = 1e-9
    val = -np.log10(x + epsilon)
    # Normalizar grosseiramente: assumindo min 1e-6 (6) e max 1 (0)
    return np.clip(val / 6.0, 0.0, 1.0)

def trans_sigmoid_cadd(x):
    """Sigmoid centrada em 20 (cutoff comum de patogenicidade)."""
    # x varia de 0 a 60+.
    return 1.0 / (1.0 + np.exp(-(x - 20.0) / 5.0))

def trans_decay(x, scale=10.0):
    """Decaimento exponencial para distâncias."""
    return np.exp(-abs(x) / scale)

def trans_minmax_norm(x, min_val, max_val):
    """Normalização linear simples."""
    val = (x - min_val) / (max_val - min_val + 1e-9)
    return np.clip(val, 0.0, 1.0)

def trans_clinvar_stars(x):
    """Normaliza estrelas (0-4) para 0-1."""
    return min(x, 4.0) / 4.0

def trans_clinvar_sig(sig_code):
    """Mapeia Pathogenic->1, Benign->0, VUS->0.5."""
    s = sig_code.lower()
    if "pathogenic" in s and "conflict" not in s: return 1.0
    if "benign" in s and "conflict" not in s: return 0.0
    return 0.5  # VUS ou Conflicting

def trans_impact_tier(tier):
    """Ordinal encoding para consequência."""
    t = tier.lower()
    if "stop" in t or "frameshift" in t: return 1.0
    if "missense" in t: return 0.6
    if "synonymous" in t: return 0.1
    return 0.3

# ==============================================================================
# 1. ESTÁGIO 1 & 2: DEDUPLICAÇÃO E FIREWALL CLÁSSICO
# ==============================================================================

if not os.path.exists(NOME_VCF_ENTRADA):
    raise FileNotFoundError(f"VCF não encontrado: {NOME_VCF_ENTRADA}")

reader = vcfpy.Reader.from_path(NOME_VCF_ENTRADA)

# Dicionário para deduplicação: Chave=ID_Variante, Valor=Dados
variantes_unicas = {}
contagem_total = 0
contagem_filtradas_ba1 = 0
contagem_filtradas_clinvar = 0

print(f"Lendo VCF: {NOME_VCF_ENTRADA} ...")

for record in reader:
    contagem_total += 1
    
    # ID Único da Variante (Chave Hash)
    var_id = f"{record.CHROM}:{record.POS}:{record.REF}:{record.ALT[0].value}"
    
    # Extração de Features Cruciais para Filtro
    info = record.INFO
    af_global = parse_float(info.get('AF_gnomAD_Global'), 0.0)
    clinvar_sig = parse_str(info.get('ClinVar_Sig_Code'))
    clinvar_stars = parse_float(info.get('ClinVar_Stars'), 0.0)
    
    # --- FIREWALL CLÁSSICO (FILTROS RÍGIDOS) ---
    
    # Regra 1: BA1 (Benigno Absoluto por Frequência)
    if af_global > LIMITE_AF_BA1:
        contagem_filtradas_ba1 += 1
        continue
        
    # Regra 2: ClinVar Benigno com Confiança (3+ estrelas)
    if "Benign" in clinvar_sig and clinvar_stars >= 3:
        contagem_filtradas_clinvar += 1
        continue
        
    # Regra 3: ClinVar Pathogenic com Confiança (3+ estrelas) - Já resolvido
    if "Pathogenic" in clinvar_sig and clinvar_stars >= 3:
        contagem_filtradas_clinvar += 1
        continue

    # Se passou no Firewall, processamos as 32 features
    # Só processamos se for a primeira vez que vemos essa variante (Deduplicação)
    if var_id not in variantes_unicas:
        
        # --- EXTRAÇÃO E TRANSFORMAÇÃO DAS 32 FEATURES ---
        # Vetor bruto
        feats = []
        
        # 1. AF_gnomAD_Global (Log Invert)
        feats.append(trans_log_invert(af_global))
        
        # 2. AF_gnomAD_PopMax (Log Invert)
        af_popmax = parse_float(info.get('AF_gnomAD_PopMax'), 0.0)
        feats.append(trans_log_invert(af_popmax))
        
        # 3. REVEL_Score (Raw 0-1)
        feats.append(parse_float(info.get('REVEL_Score'), 0.5)) # Imputa média se null
        
        # 4. SpliceAI_DeltaMax (Raw 0-1)
        feats.append(parse_float(info.get('SpliceAI_DeltaMax'), 0.0))
        
        # 5. CADD_PHRED (Sigmoid)
        feats.append(trans_sigmoid_cadd(parse_float(info.get('CADD_PHRED'), 0.0)))
        
        # 6. PhyloP_100way (StandardScaler-ish: range -4 a 10) -> MinMax mapping
        pp = parse_float(info.get('PhyloP_100way'), 0.0)
        feats.append(trans_minmax_norm(pp, -4.0, 10.0))
        
        # 7. ClinVar_Stars (Norm 0-1)
        feats.append(trans_clinvar_stars(clinvar_stars))
        
        # 8. ClinVar_Sig_Code (Ordinal)
        feats.append(trans_clinvar_sig(clinvar_sig))
        
        # 9. pLI_Score (Raw 0-1)
        feats.append(parse_float(info.get('pLI_Score'), 0.0))
        
        # 10. Z_Score_Missense (Sigmoid centrada em 0)
        z_score = parse_float(info.get('Z_Score_Missense'), 0.0)
        feats.append(1.0 / (1.0 + np.exp(-z_score)))
        
        # 11. Consequence_Tier (Ordinal)
        feats.append(trans_impact_tier(parse_str(info.get('Consequence_Tier'))))
        
        # 12. Dist_Splice_Junction (Decay)
        feats.append(trans_decay(parse_float(info.get('Dist_Splice_Junction'), 1000.0)))
        
        # 13. AlphaFold_pLDDT (Invert Norm: instável 0 -> estruturado 1? Não, prompt diz estruturado é pior se mutar)
        # Tabela: (100-x)/100 ? Não, tabela diz núcleos rígidos (pLDDT alto) são piores.
        # Vamos manter: pLDDT alto (100) -> Valor alto (1.0).
        feats.append(parse_float(info.get('AlphaFold_pLDDT'), 0.0) / 100.0)
        
        # 14. Pfam_Domain (Binary)
        dom = parse_str(info.get('Pfam_Domain'))
        feats.append(1.0 if dom != "." and dom != "" else 0.0)
        
        # 15. Mastermind_Counts (Log Scale)
        counts = parse_float(info.get('Mastermind_Counts'), 0.0)
        feats.append(np.clip(np.log1p(counts) / 5.0, 0.0, 1.0)) # log(100) ~ 4.6
        
        # 16. HPO_Term_Match (Raw)
        feats.append(parse_float(info.get('HPO_Term_Match'), 0.0))
        
        # 17. AC_Hom (Invert: se alto, benigno)
        ac_hom = parse_float(info.get('AC_Hom'), 0.0)
        feats.append(trans_decay(ac_hom, scale=5.0)) # Se hom>0 cai rapido
        
        # 18. Gene_Haploinsufficiency (Ordinal Norm 0-3)
        gh = parse_float(info.get('Gene_Haploinsufficiency'), 0.0)
        feats.append(gh / 3.0)
        
        # 19. BayesDel_noAF (MinMax -1 a 1 -> 0 a 1)
        bd = parse_float(info.get('BayesDel_noAF'), 0.0)
        feats.append(trans_minmax_norm(bd, -1.0, 1.0))
        
        # 20. GenoCanyon (Raw)
        feats.append(parse_float(info.get('GenoCanyon'), 0.5))
        
        # 21. Eigen-PCAD (MinMax approx 0 a 20)
        eig = parse_float(info.get('Eigen-PCAD'), 0.0)
        feats.append(trans_minmax_norm(eig, 0.0, 20.0))
        
        # 22. GERP++_RS (MinMax -12 a 6)
        gerp = parse_float(info.get('GERP++_RS'), 0.0)
        feats.append(trans_minmax_norm(gerp, -12.0, 6.0))
        
        # 23. LoF_Tool_Score (Raw, low is bad -> Invert?)
        # Tabela diz: percentil baixo = ruim. Para VUS patogênica queremos valor ALTO.
        # Entao invertemos: 1 - score.
        lof = parse_float(info.get('LoF_Tool_Score'), 1.0)
        feats.append(1.0 - lof)
        
        # 24. SIFT_score (Invertido: 1 - score)
        # SIFT original: 0=Deletério (Bad), 1=Tolerado (Good).
        # QML target: 1=Deletério (Bad), 0=Tolerado (Good).
        sift_val = parse_float(info.get('SIFT_score'), 1.0) # Default 1.0 (Benigno)
        feats.append(1.0 - sift_val)
        
        # 25. PolyPhen2_HVAR (Raw Prob D)
        feats.append(parse_float(info.get('PolyPhen2_HVAR'), 0.0))
        
        # 26. InterVar_Auto (Ordinal)
        ivar = parse_str(info.get('InterVar_Auto'))
        feats.append(trans_clinvar_sig(ivar)) # Reutiliza logica Path/Benign
        
        # 27. Num_Path_ClinVar (Log)
        n_path = parse_float(info.get('Num_Path_ClinVar'), 0.0)
        feats.append(np.clip(np.log1p(n_path) / 5.0, 0.0, 1.0))
        
        # 28. Num_Benign_ClinVar (Log Invert - Benigno reduz score)
        n_ben = parse_float(info.get('Num_Benign_ClinVar'), 0.0)
        # Score final deve ser alto para patogenico. Se tem muito benigno, score deve ser baixo.
        # Decay function aqui serve bem.
        feats.append(trans_decay(n_ben, scale=2.0))
        
        # 29. Is_Canonical (Binary)
        can = parse_str(info.get('Is_Canonical'))
        feats.append(1.0 if can in ['YES', '1', 'True'] else 0.0)
        
        # 30. B_Statistic (MinMax 0-1000)
        bstat = parse_float(info.get('B_Statistic'), 0.0)
        feats.append(trans_minmax_norm(bstat, 0.0, 1000.0))
        
        # 31. Distance_to_Domain (Decay)
        dist_dom = parse_float(info.get('Distance_to_Domain'), 1000.0)
        feats.append(trans_decay(dist_dom, scale=50.0))
        
        # 32. Repetitive_Region (Invert: Região repetitiva é ruim tecnicamente -> score baixo/confiança baixa?)
        # Tabela: "Variantes em regioes repetitivas podem ser artefatos (peso negativo)"
        # Se for repetitiva (1), queremos que o vetor 'enfraqueça' ou tenha uma flag.
        # Vamos codificar como: 0 = Repetitiva (Ruim), 1 = Única (Bom/Confiável).
        rep = parse_str(info.get('Repetitive_Region'))
        feats.append(0.0 if rep == '1' else 1.0)
        
        # Guardar variante processada
        variantes_unicas[var_id] = {
            'features': np.array(feats, dtype=np.float32),
            'samples': [] # Lista de pacientes que têm essa variante
        }

    # Adicionar o paciente atual à lista da variante (para retorno reverso depois)
    # No VCFpy, calls tem as amostras.
    for call in record.calls:
        if call.called and call.gt_type != 0: # 0=HomRef, 1=Het, 2=HomAlt
            variantes_unicas[var_id]['samples'].append(call.sample)

print(f"Total Variantes Lidas: {contagem_total}")
print(f"Filtradas pelo Firewall Clássico (BA1/Freq): {contagem_filtradas_ba1}")
print(f"Filtradas pelo Firewall Clássico (ClinVar Res): {contagem_filtradas_clinvar}")
print(f"Variantes Únicas Restantes (Pool): {len(variantes_unicas)}")

# ==============================================================================
# 2. ESTÁGIO 3: SELEÇÃO POR ENTROPIA (AMBIGUIDADE)
# ==============================================================================

print("\nCalculando Entropia e Selecionando Candidatas para Quantum...")

candidatas = []

for vid, data in variantes_unicas.items():
    feats = data['features']
    
    # Cálculo da Entropia / Discordância
    # Se os preditores concordam (todos altos ou todos baixos), desvio padrão é baixo.
    # Se discordam (uns 0, outros 1), desvio padrão é alto.
    discordancia = np.std(feats)
    
    # Também podemos usar a média para filtrar coisas muito benignas que passaram pelo firewall
    media_score = np.mean(feats)
    
    # Regra de Seleção:
    # 1. Discordância alta (ambiguidade)
    # OU
    # 2. Score médio em "zona cinza" (0.3 a 0.7)
    score_prioridade = discordancia
    
    if 0.3 < media_score < 0.7:
        score_prioridade += 0.1 # Bonus para zona cinza
        
    candidatas.append({
        'id': vid,
        'features': feats,
        'discordancia': discordancia,
        'score_final': score_prioridade,
        'samples_count': len(data['samples'])
    })

# Ordenar por maior discordância (As mais difíceis primeiro)
candidatas.sort(key=lambda x: x['score_final'], reverse=True)

# Corte do Top N
selecionadas = candidatas[:TOP_N_SELECAO]
print(f"Variantes selecionadas para o QPU: {len(selecionadas)} (Top {TOP_N_SELECAO})")
if len(selecionadas) > 0:
    print(f"Maior Discordância: {selecionadas[0]['discordancia']:.4f}")
    print(f"Menor Discordância (Corte): {selecionadas[-1]['discordancia']:.4f}")

# ==============================================================================
# 3. PREPARAÇÃO PARA AMPLITUDE EMBEDDING (NORMALIZAÇÃO L2)
# ==============================================================================

X_final = []
mapa_variantes = []

for item in selecionadas:
    vec = item['features']
    
    # Normalização L2 (Obrigatória para Amplitude Embedding)
    # O vetor deve ter norma 1: sqrt(sum(x^2)) = 1
    norma = np.linalg.norm(vec)
    if norma > 0:
        vec_normalized = vec / norma
    else:
        vec_normalized = vec # Caso vetor seja tudo zero (raro)
        
    X_final.append(vec_normalized)
    mapa_variantes.append({
        'Variant_ID': item['id'],
        'Discordancia': item['discordancia'],
        'Num_Pacientes_Afetados': item['samples_count']
    })

X_quantum = np.array(X_final, dtype=np.float32)

# Salvar Arquivos
np.save(NOME_X_QUANTUM, X_quantum)
pd.DataFrame(mapa_variantes).to_csv(NOME_MAP_SAIDA, index=False)

print("\n--- PRÉ-PROCESSAMENTO CONCLUÍDO ---")
print(f"Matriz Quântica Salva: {NOME_X_QUANTUM}")
print(f"Shape: {X_quantum.shape} (Variantes x 32 Features)")
print(f"Mapeamento Salvo: {NOME_MAP_SAIDA}")
print("Pronto para injetar no pipeline 'AmplitudeEmbedding'.")