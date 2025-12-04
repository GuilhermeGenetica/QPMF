import vcfpy
import random
import numpy as np
import pandas as pd
import datetime
import sys
import os

# --- Configuração de Caminhos Absolutos ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

N_PACIENTES = 500
N_VARIANTES_TOTAL = 5000
N_VARIANTES_MONOGENICAS = 5
N_VARIANTES_POLIGENICAS = 300
PREVALENCIA_DOENCA = 0.05

PROBABILIDADE_LD_BLOCK = 0.3
TAMANHO_MEDIO_BLOCK_LD = 5
N_PARES_EPISTASIA = 15

# Nomes dos ficheiros com caminho completo para evitar erros de endereçamento
NOME_VCF_SAIDA = os.path.join(BASE_DIR, 'dados_geneticos.vcf')
NOME_CSV_SAIDA = os.path.join(BASE_DIR, 'fenotipos.csv')

random.seed(42)
np.random.seed(42)

print(f"--- SIMULADOR GENÓMICO de VCF: LD BLOCKS + EPISTASIA (32 Features Annotations) ---")
print(f"População: {N_PACIENTES} | Variantes: {N_VARIANTES_TOTAL}")

NOMES_AMOSTRAS = [f"PACIENTE_{i:04d}" for i in range(1, N_PACIENTES + 1)]
CHROMOSSOMAS = [f"chr{i}" for i in range(1, 23)]
ALLELES = ['A', 'C', 'G', 'T']

indices = np.arange(N_VARIANTES_TOTAL)
np.random.shuffle(indices)

idxs_drivers_monogenicos = set(indices[:N_VARIANTES_MONOGENICAS])
idxs_risco_poligenico = set(indices[N_VARIANTES_MONOGENICAS : N_VARIANTES_MONOGENICAS + N_VARIANTES_POLIGENICAS])

pesos_reais = np.zeros(N_VARIANTES_TOTAL)
pesos_reais[list(idxs_drivers_monogenicos)] = 8.0 
pesos_reais[list(idxs_risco_poligenico)] = np.random.normal(0.3, 0.1, N_VARIANTES_POLIGENICAS)

mafs_globais = np.random.beta(1, 10, N_VARIANTES_TOTAL)
mafs_globais[list(idxs_drivers_monogenicos)] = np.random.uniform(0.00005, 0.005, N_VARIANTES_MONOGENICAS)
mafs_globais[list(idxs_risco_poligenico)] = np.random.uniform(0.05, 0.5, N_VARIANTES_POLIGENICAS)

matriz_genotipos = np.zeros((N_PACIENTES, N_VARIANTES_TOTAL), dtype=int)
remaining_in_block = 0
previous_variant_idx = -1
ld_block_count = 0

print("Gerando genótipos com Desequilíbrio de Ligação (LD)...")

for v in range(N_VARIANTES_TOTAL):
    if remaining_in_block > 0:
        base_gt = matriz_genotipos[:, previous_variant_idx]
        noise = np.random.choice([0, 1], size=N_PACIENTES, p=[0.95, 0.05])
        new_gt = np.where(noise == 1, np.random.choice([0,1,2], N_PACIENTES), base_gt)
        matriz_genotipos[:, v] = new_gt
        remaining_in_block -= 1
    else:
        p = 1 - mafs_globais[v]
        q = mafs_globais[v]
        probs = [p**2, 2*p*q, q**2]
        probs = np.array(probs) / np.sum(probs)
        matriz_genotipos[:, v] = np.random.choice([0, 1, 2], size=N_PACIENTES, p=probs)
        
        if random.random() < PROBABILIDADE_LD_BLOCK:
            remaining_in_block = int(np.random.exponential(TAMANHO_MEDIO_BLOCK_LD))
            ld_block_count += 1
            previous_variant_idx = v

print(f"Blocos LD gerados: {ld_block_count}")

def calcular_acmg(func, exonic_func, maf, cadd, revel, clinvar_sig_sim):
    criteria = []
    
    try: maf_val = float(maf)
    except: maf_val = 0.0

    if maf_val < 0.0001: criteria.append("PM2") 
    elif maf_val > 0.05: criteria.append("BA1") 
    elif maf_val > 0.01: criteria.append("BS1") 

    if exonic_func in ["stopgain", "frameshift_insertion", "frameshift_deletion", "stop_gained"]:
        if "BA1" not in criteria and "BS1" not in criteria:
            criteria.append("PVS1")

    try:
        cadd_val = float(cadd) if cadd != "." else 0
        revel_val = float(revel) if revel != "." else 0
        if cadd_val > 25 or revel_val > 0.75: criteria.append("PP3")
        elif cadd_val < 10 and revel_val < 0.2: criteria.append("BP4")
    except: pass

    if clinvar_sig_sim == "Pathogenic":
        if "PVS1" not in criteria: criteria.append("PS1") 
        if "PM2" in criteria: criteria.append("PM1")
        
    score = 0
    for c in criteria:
        if c.startswith("PVS"): score += 8
        elif c.startswith("PS"): score += 4
        elif c.startswith("PM"): score += 2
        elif c.startswith("PP"): score += 1
        elif c.startswith("BA"): score -= 8
        elif c.startswith("BS"): score -= 4
        elif c.startswith("BP"): score -= 1
    
    if score >= 10: acmg_class = "Pathogenic"
    elif score >= 6: acmg_class = "Likely_pathogenic"
    elif score >= 0: acmg_class = "Uncertain_significance"
    elif score >= -5: acmg_class = "Likely_benign"
    else: acmg_class = "Benign"
    
    return acmg_class, ",".join(criteria)

def calcular_abc(acmg_criteria, is_driver, is_polygenic, phenotype_match=True):
    crit_list = acmg_criteria.split(",") if acmg_criteria else []
    
    step_a = 0
    if "PVS1" in crit_list or "PS1" in crit_list: step_a = 5
    elif any(c in crit_list for c in ["PM4", "PM5"]): step_a = 4
    elif any(c in crit_list for c in ["PM1", "PM2", "PP3"]): step_a = 3
    elif "BA1" in crit_list: step_a = 1
    elif "BS1" in crit_list or "BP4" in crit_list: step_a = 2
    else: step_a = 0 

    step_b = 0
    if is_driver: step_b = random.choice([3, 4, 5]) 
    elif is_polygenic: step_b = 2
    elif step_a >= 3 and phenotype_match: step_b = 1 
    else: step_b = 0

    total_score = step_a + step_b
    if total_score >= 10: abc_class = "A"
    elif total_score == 9: abc_class = "B"
    elif total_score == 8: abc_class = "C"
    elif total_score >= 6: abc_class = "D"
    elif total_score >= 4: abc_class = "E"
    elif total_score == 3: abc_class = "F"
    else: abc_class = "0"
    
    return step_a, step_b, abc_class

def gerar_anotacoes_completas(context, gene_symbol, ref, alt):
    """
    Gera as 32 features solicitadas com aleatoriedade realistica baseada no contexto:
    context: 'P' (Patogênico), 'B' (Benigno), 'V' (VUS)
    """
    vals = {}
    
    # --- 1. AF_gnomAD_Global ---
    if context == 'P': vals['AF_gnomAD_Global'] = "{:.7f}".format(random.uniform(0.000001, 0.00005))
    elif context == 'B': vals['AF_gnomAD_Global'] = "{:.4f}".format(np.random.beta(2, 5)) # Pode ser alto
    else: # VUS
        if random.random() < 0.3: vals['AF_gnomAD_Global'] = "."
        else: vals['AF_gnomAD_Global'] = "{:.6f}".format(random.uniform(0.00001, 0.0005))

    # --- 2. AF_gnomAD_PopMax ---
    if context == 'P': vals['AF_gnomAD_PopMax'] = "{:.7f}".format(random.uniform(0.000001, 0.0001))
    elif context == 'B': vals['AF_gnomAD_PopMax'] = "{:.4f}".format(np.random.beta(2, 4))
    else: vals['AF_gnomAD_PopMax'] = "{:.5f}".format(random.uniform(0.00005, 0.001))

    # --- 3. REVEL_Score ---
    if context == 'P': vals['REVEL_Score'] = "{:.3f}".format(np.random.beta(15, 2)) # Skewed to 1
    elif context == 'B': vals['REVEL_Score'] = "{:.3f}".format(np.random.beta(1, 10)) # Skewed to 0
    else: vals['REVEL_Score'] = "{:.3f}".format(random.uniform(0.35, 0.65)) # Zona Cinza

    # --- 4. SpliceAI_DeltaMax ---
    if context == 'P' and random.random() < 0.3: vals['SpliceAI_DeltaMax'] = "{:.2f}".format(random.uniform(0.8, 1.0))
    elif context == 'B': vals['SpliceAI_DeltaMax'] = "{:.2f}".format(random.uniform(0.0, 0.05))
    else: vals['SpliceAI_DeltaMax'] = "{:.2f}".format(random.uniform(0.05, 0.30))

    # --- 5. CADD_PHRED ---
    if context == 'P': vals['CADD_PHRED'] = "{:.1f}".format(np.random.normal(30, 4))
    elif context == 'B': vals['CADD_PHRED'] = "{:.1f}".format(np.random.normal(5, 3))
    else: vals['CADD_PHRED'] = "{:.1f}".format(random.uniform(15.0, 22.0))

    # --- 6. PhyloP_100way ---
    if context == 'P': vals['PhyloP_100way'] = "{:.2f}".format(random.uniform(4.0, 9.0))
    elif context == 'B': vals['PhyloP_100way'] = "{:.2f}".format(random.uniform(-3.0, 0.5))
    else: vals['PhyloP_100way'] = "{:.2f}".format(random.uniform(0.5, 2.5))

    # --- 7. ClinVar_Stars ---
    if context == 'P': vals['ClinVar_Stars'] = str(random.choice([1, 2, 3]))
    elif context == 'B': vals['ClinVar_Stars'] = str(random.choice([2, 3]))
    else: vals['ClinVar_Stars'] = "0"

    # --- 8. ClinVar_Sig_Code ---
    if context == 'P': vals['ClinVar_Sig_Code'] = "Pathogenic"
    elif context == 'B': vals['ClinVar_Sig_Code'] = "Benign"
    else: vals['ClinVar_Sig_Code'] = "Uncertain_significance"

    # --- 9. pLI_Score ---
    if context == 'P': vals['pLI_Score'] = "{:.2f}".format(random.uniform(0.90, 1.00))
    elif context == 'B': vals['pLI_Score'] = "{:.2f}".format(random.uniform(0.00, 0.10))
    else: vals['pLI_Score'] = "{:.2f}".format(random.uniform(0.40, 0.80))

    # --- 10. Z_Score_Missense ---
    if context == 'P': vals['Z_Score_Missense'] = "{:.2f}".format(random.uniform(3.0, 5.0))
    elif context == 'B': vals['Z_Score_Missense'] = "{:.2f}".format(random.uniform(-1.5, 0.5))
    else: vals['Z_Score_Missense'] = "{:.2f}".format(random.uniform(1.0, 2.5))

    # --- 11. Consequence_Tier ---
    if context == 'P': vals['Consequence_Tier'] = random.choice(["stop_gained", "frameshift_variant", "missense_variant"])
    elif context == 'B': vals['Consequence_Tier'] = "synonymous_variant"
    else: vals['Consequence_Tier'] = "missense_variant"

    # --- 12. Dist_Splice_Junction ---
    if context == 'P' and "splice" in vals.get('SpliceAI_DeltaMax', '0'): 
         vals['Dist_Splice_Junction'] = str(random.randint(1, 5))
    elif context == 'B': vals['Dist_Splice_Junction'] = str(random.randint(50, 200))
    else: vals['Dist_Splice_Junction'] = str(random.randint(4, 20))

    # --- 13. AlphaFold_pLDDT ---
    if context == 'P': vals['AlphaFold_pLDDT'] = "{:.1f}".format(random.uniform(85, 98))
    elif context == 'B': vals['AlphaFold_pLDDT'] = "{:.1f}".format(random.uniform(30, 60))
    else: vals['AlphaFold_pLDDT'] = "{:.1f}".format(random.uniform(60, 80))

    # --- 14. Pfam_Domain ---
    if context == 'P': vals['Pfam_Domain'] = "Pkinase_Tyr"
    elif context == 'B': vals['Pfam_Domain'] = "."
    else: vals['Pfam_Domain'] = "SH3_1" if random.random() > 0.5 else "Domain_Unknown"

    # --- 15. Mastermind_Counts ---
    if context == 'P': vals['Mastermind_Counts'] = str(random.randint(20, 300))
    elif context == 'B': vals['Mastermind_Counts'] = str(random.randint(0, 5))
    else: vals['Mastermind_Counts'] = "0"

    # --- 16. HPO_Term_Match ---
    if context == 'P': vals['HPO_Term_Match'] = "{:.2f}".format(random.uniform(0.8, 1.0))
    elif context == 'B': vals['HPO_Term_Match'] = "{:.2f}".format(random.uniform(0.0, 0.2))
    else: vals['HPO_Term_Match'] = "{:.2f}".format(random.uniform(0.3, 0.6))

    # --- 17. AC_Hom ---
    if context == 'P': vals['AC_Hom'] = "0"
    elif context == 'B': vals['AC_Hom'] = str(random.randint(100, 2000))
    else: vals['AC_Hom'] = "0"

    # --- 18. Gene_Haploinsufficiency ---
    if context == 'P': vals['Gene_Haploinsufficiency'] = "3"
    elif context == 'B': vals['Gene_Haploinsufficiency'] = "0"
    else: vals['Gene_Haploinsufficiency'] = "1"

    # --- 19. BayesDel_noAF ---
    if context == 'P': vals['BayesDel_noAF'] = "{:.2f}".format(random.uniform(0.5, 0.7))
    elif context == 'B': vals['BayesDel_noAF'] = "{:.2f}".format(random.uniform(-0.9, -0.4))
    else: vals['BayesDel_noAF'] = "{:.2f}".format(random.uniform(0.1, 0.2))

    # --- 20. GenoCanyon ---
    if context == 'P': vals['GenoCanyon'] = "{:.2f}".format(random.uniform(0.95, 1.0))
    elif context == 'B': vals['GenoCanyon'] = "{:.2f}".format(random.uniform(0.0, 0.1))
    else: vals['GenoCanyon'] = "{:.2f}".format(random.uniform(0.4, 0.6))

    # --- 21. Eigen-PCAD ---
    if context == 'P': vals['Eigen-PCAD'] = "{:.1f}".format(random.uniform(7.0, 10.0))
    elif context == 'B': vals['Eigen-PCAD'] = "{:.1f}".format(random.uniform(0.0, 2.0))
    else: vals['Eigen-PCAD'] = "{:.1f}".format(random.uniform(3.0, 5.0))

    # --- 22. GERP++_RS ---
    if context == 'P': vals['GERP++_RS'] = "{:.2f}".format(random.uniform(4.0, 6.0))
    elif context == 'B': vals['GERP++_RS'] = "{:.2f}".format(random.uniform(-4.0, 0.0))
    else: vals['GERP++_RS'] = "{:.2f}".format(random.uniform(1.5, 3.0))

    # --- 23. LoF_Tool_Score ---
    # Percentil baixo = ruim (intolerante)
    if context == 'P': vals['LoF_Tool_Score'] = "{:.2f}".format(random.uniform(0.0, 0.1))
    elif context == 'B': vals['LoF_Tool_Score'] = "{:.2f}".format(random.uniform(0.7, 1.0))
    else: vals['LoF_Tool_Score'] = "{:.2f}".format(random.uniform(0.25, 0.45))

    # --- 24. SIFT_score (Continuo) ---
    if context == 'P': vals['SIFT_score'] = "{:.3f}".format(random.uniform(0.00, 0.04)) # Deletério (<0.05)
    elif context == 'B': vals['SIFT_score'] = "{:.3f}".format(random.uniform(0.06, 1.00)) # Tolerado
    else: vals['SIFT_score'] = "{:.3f}".format(random.uniform(0.00, 1.00)) # Incerto/Amplo

    # --- 25. PolyPhen2_HVAR ---
    if context == 'P': vals['PolyPhen2_HVAR'] = "{:.3f}".format(random.uniform(0.9, 1.0))
    elif context == 'B': vals['PolyPhen2_HVAR'] = "{:.3f}".format(random.uniform(0.0, 0.1))
    else: vals['PolyPhen2_HVAR'] = "{:.3f}".format(random.uniform(0.5, 0.8))

    # --- 26. InterVar_Auto ---
    if context == 'P': vals['InterVar_Auto'] = "Pathogenic"
    elif context == 'B': vals['InterVar_Auto'] = "Benign"
    else: vals['InterVar_Auto'] = "Uncertain"

    # --- 27. Num_Path_ClinVar ---
    if context == 'P': vals['Num_Path_ClinVar'] = str(random.randint(10, 50))
    elif context == 'B': vals['Num_Path_ClinVar'] = "0"
    else: vals['Num_Path_ClinVar'] = "1"

    # --- 28. Num_Benign_ClinVar ---
    if context == 'P': vals['Num_Benign_ClinVar'] = "0"
    elif context == 'B': vals['Num_Benign_ClinVar'] = str(random.randint(5, 30))
    else: vals['Num_Benign_ClinVar'] = "1"

    # --- 29. Is_Canonical ---
    if context == 'P': vals['Is_Canonical'] = "YES"
    elif context == 'B': vals['Is_Canonical'] = "NO" if random.random() < 0.3 else "YES"
    else: vals['Is_Canonical'] = "YES"

    # --- 30. B_Statistic ---
    if context == 'P': vals['B_Statistic'] = str(random.randint(900, 1000))
    elif context == 'B': vals['B_Statistic'] = str(random.randint(0, 200))
    else: vals['B_Statistic'] = str(random.randint(400, 600))

    # --- 31. Distance_to_Domain ---
    if context == 'P': vals['Distance_to_Domain'] = "0"
    elif context == 'B': vals['Distance_to_Domain'] = str(random.randint(200, 1000))
    else: vals['Distance_to_Domain'] = str(random.randint(1, 10))

    # --- 32. Repetitive_Region ---
    if context == 'P': vals['Repetitive_Region'] = "0"
    elif context == 'B': vals['Repetitive_Region'] = "1"
    else: vals['Repetitive_Region'] = "1" if random.random() < 0.5 else "0"

    # --- Mapping Legacy Fields for System Compatibility ---
    # ACMG Calculator expects: Func.refGene, ExonicFunc.refGene, gnomAD_genome_ALL, CADD_phred, REVEL_score
    vals['Func.refGene'] = "exonic" if context in ['P', 'V'] else "intronic"
    vals['ExonicFunc.refGene'] = vals['Consequence_Tier'] # Mapping
    vals['AAChange.refGene'] = f"{gene_symbol}:NM_001:exon1:c.100{ref}>{alt}:p.Sim"
    vals['gnomAD_genome_ALL'] = vals['AF_gnomAD_Global'] # Mapping
    vals['CADD_phred'] = vals['CADD_PHRED'] # Mapping
    vals['REVEL_score'] = vals['REVEL_Score'] # Mapping

    return vals

header = vcfpy.Header()
header.add_line(vcfpy.HeaderLine('fileformat', 'VCFv4.2'))
header.add_line(vcfpy.HeaderLine('source', 'Simulador_V7_Ultra_Realista'))
header.add_line(vcfpy.HeaderLine('reference', 'GRCh38'))
for c in CHROMOSSOMAS: header.add_contig_line({'ID': c, 'length': '248956422'})

# --- Definição das 32 Features + Legacy no Header ---
features_info = [
    # Legacy / Sistema
    ('CLNSIG', '.', 'String', 'ClinVar significance (Legacy)'),
    ('CLNDN', '.', 'String', 'Disease name'),
    ('GENEINFO', '1', 'String', 'Gene:ID'),
    ('RS', '.', 'String', 'dbSNP ID'),
    ('ACMG_Class', '1', 'String', 'Calculated ACMG Classification'),
    ('ACMG_Criteria', '.', 'String', 'ACMG Criteria Met'),
    ('ABC_Score', '1', 'String', 'ABC Score'),
    ('ABC_Class', '1', 'String', 'ABC Classification'),
    ('Func.refGene', '.', 'String', 'Genic region (Legacy)'),
    ('ExonicFunc.refGene', '.', 'String', 'Exonic func (Legacy)'),
    ('AAChange.refGene', '.', 'String', 'AA Change (Legacy)'),
    ('gnomAD_genome_ALL', '1', 'String', 'MAF (Legacy)'),
    ('CADD_phred', '1', 'String', 'CADD Score (Legacy)'),
    ('REVEL_score', '1', 'String', 'REVEL Score (Legacy)'),

    # As 32 Features Novas
    ('AF_gnomAD_Global', '1', 'String', 'Frequência alélica global'),
    ('AF_gnomAD_PopMax', '1', 'String', 'Frequência na sub-população mais frequente'),
    ('REVEL_Score', '1', 'String', 'Meta-preditor missense'),
    ('SpliceAI_DeltaMax', '1', 'String', 'Score splicing'),
    ('CADD_PHRED', '1', 'String', 'Score deleteriocidade'),
    ('PhyloP_100way', '1', 'String', 'Conservação evolutiva'),
    ('ClinVar_Stars', '1', 'String', 'Nível de revisão ClinVar'),
    ('ClinVar_Sig_Code', '1', 'String', 'Classificação ClinVar'),
    ('pLI_Score', '1', 'String', 'Intolerância LoF'),
    ('Z_Score_Missense', '1', 'String', 'Intolerância Missense'),
    ('Consequence_Tier', '1', 'String', 'Impacto'),
    ('Dist_Splice_Junction', '1', 'String', 'Distância exon (bp)'),
    ('AlphaFold_pLDDT', '1', 'String', 'Confiança estrutura 3D'),
    ('Pfam_Domain', '1', 'String', 'Domínio funcional'),
    ('Mastermind_Counts', '1', 'String', 'Artigos citados'),
    ('HPO_Term_Match', '1', 'String', 'Score fenotípico'),
    ('AC_Hom', '1', 'String', 'Contagem homozigotos'),
    ('Gene_Haploinsufficiency', '1', 'String', 'Score haploinsuficiência'),
    ('BayesDel_noAF', '1', 'String', 'Meta-preditor Bayesiano'),
    ('GenoCanyon', '1', 'String', 'Score funcional integrado'),
    ('Eigen-PCAD', '1', 'String', 'Score espectral'),
    ('GERP++_RS', '1', 'String', 'Score rejeição evolutiva'),
    ('LoF_Tool_Score', '1', 'String', 'Percentil intolerância LoF'),
    ('SIFT_score', '1', 'Float', 'Score SIFT (0-1, baixo=deletério)'),
    ('PolyPhen2_HVAR', '1', 'String', 'Previsão PolyPhen2'),
    ('InterVar_Auto', '1', 'String', 'Classificação Auto InterVar'),
    ('Num_Path_ClinVar', '1', 'String', 'Submissões Pathogenic'),
    ('Num_Benign_ClinVar', '1', 'String', 'Submissões Benign'),
    ('Is_Canonical', '1', 'String', 'Transcrito canônico'),
    ('B_Statistic', '1', 'String', 'Conservação fundo'),
    ('Distance_to_Domain', '1', 'String', 'Distância domínio funcional'),
    ('Repetitive_Region', '1', 'String', 'Região repetitiva')
]

for i in features_info: header.add_info_line({'ID': i[0], 'Number': i[1], 'Type': i[2], 'Description': i[3]})

formats = [('GT','1','String'), ('AD','R','Integer'), ('DP','1','Integer'), ('GQ','1','Integer'), ('PL','G','Integer')]
for f in formats: header.add_format_line({'ID': f[0], 'Number': f[1], 'Type': f[2], 'Description': '...'})

header.samples = vcfpy.SamplesInfos(NOMES_AMOSTRAS)

registos = []
print("Escrevendo VCF...")

for v_idx in range(N_VARIANTES_TOTAL):
    chrom = CHROMOSSOMAS[v_idx % 22]
    pos = 10000 + (v_idx * 100)
    ref = random.choice(ALLELES)
    alt = random.choice([a for a in ALLELES if a != ref])
    gene_id = (v_idx % 1000) + 1
    gene_symbol = f"GENE_{gene_id}"
    
    is_driver = v_idx in idxs_drivers_monogenicos
    is_polygenic = v_idx in idxs_risco_poligenico
    
    # Determina contexto para geração de features
    if is_driver: context = 'P'
    elif is_polygenic: context = 'V'
    else: context = 'B'
    
    # Gera todas as 32 features + legacy
    annovar = gerar_anotacoes_completas(context, gene_symbol, ref, alt)
    
    # Compatibilidade com lógica antiga
    clnsig_sim = annovar['ClinVar_Sig_Code']
    
    acmg_class, acmg_crit = calcular_acmg(
        annovar["Func.refGene"], annovar["ExonicFunc.refGene"], 
        annovar["gnomAD_genome_ALL"], annovar["CADD_phred"], 
        annovar["REVEL_score"], clnsig_sim
    )
    
    pheno_match = True if (is_driver or is_polygenic) else (random.random() < 0.1)
    abc_a, abc_b, abc_c = calcular_abc(acmg_crit, is_driver, is_polygenic, pheno_match)
    
    # Monta o dicionário INFO final (merge)
    info = {
        "CLNSIG": [clnsig_sim], 
        "CLNDN": ["Cardiomyopathy" if (is_driver or is_polygenic) else "not_specified"], 
        "GENEINFO": f"{gene_symbol}:{gene_id}", 
        "RS": [f"rs{random.randint(1000,999999)}"], 
        "ACMG_Class": acmg_class, 
        "ACMG_Criteria": [acmg_crit] if acmg_crit else ["."], 
        "ABC_Score": f"A:{abc_a}+B:{abc_b}",
        "ABC_Class": abc_c
    }
    # Adiciona todas as anotações geradas ao INFO
    info.update(annovar)
    # Garante que campos legacy sejam listas se necessário pelo VCFpy (Strings geralmente OK)
    # Ajuste fino: Alguns campos legacy originais eram listas, vamos manter compatibilidade se precisar.
    # O código original usava listas para Func.refGene, etc.
    legacy_list_fields = ["Func.refGene", "ExonicFunc.refGene", "AAChange.refGene"]
    for f in legacy_list_fields:
        if f in info and not isinstance(info[f], list):
            info[f] = [info[f]]

    calls = []
    for s_idx in range(N_PACIENTES):
        gt_val = matriz_genotipos[s_idx, v_idx]
        gt_str = "0/0" if gt_val == 0 else ("0/1" if gt_val == 1 else "1/1")
        calls.append(vcfpy.Call(sample=NOMES_AMOSTRAS[s_idx], data={"GT": gt_str, "AD": [30, 0], "DP": 30, "GQ": 99, "PL": [0,10,100]}))

    registos.append(vcfpy.Record(
        CHROM=chrom, POS=pos, ID=[info['RS'][0]], REF=ref, ALT=[vcfpy.Substitution('SNV', alt)],
        QUAL=999, FILTER=['PASS'], INFO=info, FORMAT=['GT','AD','DP','GQ','PL'], calls=calls
    ))
    
    if v_idx % 500 == 0: sys.stdout.write(f"\rProgresso: {v_idx}/{N_VARIANTES_TOTAL}")

print("\nGravando arquivo VCF...")
writer = vcfpy.Writer.from_path(NOME_VCF_SAIDA, header)
for r in registos: writer.write_record(r)
writer.close()

print("Calculando fenótipos (Linear + Epistático + Ambiental)...")

score_linear = matriz_genotipos @ pesos_reais

pool_poly = list(idxs_risco_poligenico)
score_epistatico = np.zeros(N_PACIENTES)

for _ in range(N_PARES_EPISTASIA):
    idx_a = random.choice(pool_poly)
    idx_b = random.choice(pool_poly)
    ga = matriz_genotipos[:, idx_a]
    gb = matriz_genotipos[:, idx_b]
    interaction = (ga * gb) * 0.5
    score_epistatico += interaction

score_total_genetico = score_linear + score_epistatico
env_score = np.random.normal(0, np.std(score_total_genetico) * 0.6, N_PACIENTES)
total_liability = score_total_genetico + env_score

liability_z = (total_liability - np.mean(total_liability)) / np.std(total_liability)
threshold = np.percentile(liability_z, 100 * (1 - PREVALENCIA_DOENCA))
status_doenca = (liability_z > threshold).astype(int)

print(f"Indivíduos Doentes: {np.sum(status_doenca)} ({np.sum(status_doenca)/N_PACIENTES:.1%})")

df = pd.DataFrame({
    'ID_Amostra': NOMES_AMOSTRAS,
    'Status_Doenca': status_doenca,
    'Liability_Z_Score': np.round(liability_z, 4),
    'Score_Linear': np.round(score_linear, 4),
    'Score_Epistatico': np.round(score_epistatico, 4),
    'Score_Ambiental': np.round(env_score, 4)
})
df.to_csv(NOME_CSV_SAIDA, index=False)
print(f"Concluído. Arquivos gerados:\n - {NOME_VCF_SAIDA}\n - {NOME_CSV_SAIDA}")