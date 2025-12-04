# QPMF — Quantum-Classical Genomic Framework (Edição 2025)

Esta é uma suite de benchmark de alta fidelidade para classificação de **Variantes Genómicas de Significado Incerto (VUS)** utilizando arquiteturas de Quantum Machine Learning (variacionais e híbridas). Este repositório contém o pipeline completo — desde a simulação biológica de variantes (VCF) até à orquestração e comparação entre modelos quânticos avançados e um baseline clássico robusto — com monitorização de recursos e medidas de custo energético (CodeCarbon).

---

## 📋 Visão Geral Rápida

O QPMF foi concebido como uma infraestrutura modular e reprodutível para avaliar a potencial vantagem quântica em tarefas reais de Medicina de Precisão. O projeto garante consistência experimental (através de `config_comum.py` e `SEED_GLOBAL = 42`), regista utilização de recursos (CPU, RAM) e tenta estimar emissões de CO₂ (via `codecarbon`). Implementa estratégias de mitigação de problemas típicos de QML (Anti-Barren Plateau, inicialização *Cold Start*) e permite comparar múltiplos paradigmas quânticos com um conjunto clássico de referência.

---

## 🚀 Arquitetura do pipeline (fluxo de dados)

    **Ordem de execução (obrigatória para reprodução):**

1. `Gerador_VCF.py` — gera `dados_geneticos.vcf` + `fenotipos.csv` (simulação com LD e epistasia).
2. `preprocess_32_features.py` — engenharia de features, filtros ACMG/AMP, normalização L2 (para amplitude embedding).
3. `Classificador_Hibrido.py` — HQGA (Kernel Target Alignment) para seleção de 32 features (gera `X_hibrido.npy`, `y_hibrido.npy`).
4. `BENCHMARK_MASTER.py` — orquestra todo o benchmark (Modelos A–G + Baseline clássico), consolida resultados e gera relatórios.

> **Nota:** `BENCHMARK_MASTER.py` pode executar automaticamente todos os testes individuais; não é necessário rodar cada teste separadamente, salvo quando se deseja depuração ou experimentos isolados.

---


## 🧠 Modelos Implementados (Suite 2025)
```
| ID  | Nome do Modelo             | Arquitetura / Técnica chave                                                                 | Observações                                                                  |
|-----|----------------------------|---------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| A   | QNN Híbrido                | Amplitude Embedding + StronglyEntanglingLayers (3 camadas), LR Scheduler, Cold Start        | Compressão das 32 features em 5 qubits (`2^5 = 32`)                          |
| B   | Dual Kernel SVM / QSVM     | Amplitude Kernel híbrido + RBF clássico; QSVM Angle (8 qubits após PCA)                     | Combina similaridades clássicas e quânticas; parâmetro de mistura (alpha)    |
| C   | Quantum Boosting           | AdaBoost-QNN (SAMME.R) com weak learners (BasicEntangler) e weighted resampling             | Ensemble sequencial de ~15 QNNs leves                                        |
| D   | MPS / Tensor Net           | Matrix Product State, Weighted Loss; Persistent Homology; Entropia de Von Neumann           | Menor número de parâmetros; análise topológica dos erros                     |
| E   | Angle Encoding             | PCA → 8 features → Angle Encoding (rotações de fase)                                        | Mais qubits; estabilidade contra ruído                                       |
| F   | Hierarchical TN            | Tree Tensor Networks (TTN) e MERA                                                           | Modela hierarquias biológicas (genes → vias → fenótipo)                      |
| G   | QNN ICO                    | Interference Control Optimization com Ancilla qubit                                         | Controlo de interferência construtiva/destrutiva como mecanismo de decisão   |
| Ref | Clássico Robusto (Baseline)| Random Forest, SVM Linear, Gradient Boosting, KNN com PCA interno                           | Serve como barra de comparação (esperado ~77–80% em dados simulados)         |
```
---

## 📂 Estrutura de Ficheiros (resumo)

```
qpmf-framework/
├─ config_comum.py
├─ Gerador_VCF.py
├─ preprocess_32_features.py
├─ Classificador_Hibrido.py
├─ BENCHMARK_MASTER.py
├─ teste_modelo_A.py ... teste_modelo_G.py
├─ benchmark_quantico_anti_bp.py
├─ benchmark_classico_robusto.py
└─ RESULTADOS_QPMF/    (gerado na execução)
```

**Descrição rápida:**

- `config_comum.py` — parâmetros globais, paths, `SEED_GLOBAL = 42`, normalização L2.
- `Gerador_VCF.py` — simula 500 pacientes × 5000 variantes; modela LD, epistasia, variantes monogénicas/poligénicas; exporta `.vcf` e `.csv`.
- `preprocess_32_features.py` — calcula 32 features necessárias para embedding quântico e validações ACMG/AMP.
- `Classificador_Hibrido.py` — HQGA com Kernel Target Alignment; produz `X_hibrido.npy` / `y_hibrido.npy`.
- `BENCHMARK_MASTER.py` — auditoria de hardware, execução sequencial dos modelos, monitorização (CodeCarbon/simulado), consolidação dos resultados.
- `teste_modelo_[A-G].py` — implementações detalhadas e reproduzíveis de cada arquitetura.
- `benchmark_*` — scripts de teste para cenários controlados (anti-barren plateau, baseline clássico).
- `RESULTADOS_QPMF/` — pasta de saída com CSVs, gráficos e logs.

---

## 🛠️ Requisitos e Instalação

**Python recomendado:** 3.9, 3.10 ou 3.11.

Recomenda-se usar um ambiente virtual (`venv`) para isolamento.

### 1) Criar e ativar ambiente virtual

**Windows (PowerShell):**

```powershell
python -m venv venv_qpmf
.\venv_qpmf\Scripts\activate
```

**Linux / macOS (bash):**

```bash
python3 -m venv venv_qpmf
source venv_qpmf/bin/activate
```

### 2) Atualizar `pip` e instalar dependências

```bash
pip install --upgrade pip
pip install pennylane scikit-learn pandas numpy matplotlib vcfpy psutil codecarbon scipy networkx
```

> **Nota:** a instalação de `codecarbon` pode requerer permissões ou falhar em alguns ambientes (especialmente Windows). Os scripts incluem blocos `try...except` para permitir execução mesmo sem rastreio real (modo "simulado").

---

## ▶️ Guia de Execução — Passo a Passo

> **Importante:** certifique-se de que `config_comum.py` e os restantes ficheiros do repositório estão na mesma pasta antes de executar.

### 1) Geração de Dados Sintéticos

Gera os ficheiros `dados_geneticos.vcf` e `fenotipos.csv`.

```bash
python Gerador_VCF.py
```

### 2) Pré-processamento e Engenharia de Features

Gera `X_quantum_32dim.npy`, aplica normalização L2 e validações ACMG/AMP.

```bash
python preprocess_32_features.py
```

### 3) Seleção de Features (HQGA)

Seleciona as 32 features mais relevantes e gera `X_hibrido.npy` e `y_hibrido.npy`.

```bash
python Classificador_Hibrido.py
```

### 4) Execução do Benchmark Completo

Executa a auditoria, todos os modelos (A–G + Baseline), monitorização de energia e consolidação.

```bash
python BENCHMARK_MASTER.py
```

---

## 📊 Saídas Esperadas e Relatórios

Ao final da execução, a pasta `RESULTADOS_QPMF` conterá:

- `PLACAR_FINAL_GERAL.csv` — Leaderboard consolidado (Acurácia, Desvio Padrão, Consumo Energético).
- `resultados_quanticos_otimizados.csv` — Métricas detalhadas por modelo (A–G).
- `Graficos/` — PNGs com curvas de convergência (loss vs epochs), matrizes de confusão e topologia de erros (Persistent Homology do Modelo D).
- `LOG_GERAL_BENCHMARK.txt` — Log completo e auditável de toda a execução.

---

## ⚠️ Notas Técnicas e Considerações Práticas

- **Determinismo:** `SEED_GLOBAL = 42` em `config_comum.py` para reprodutibilidade experimental.
- **Monitorização:** uso de `psutil` (CPU/RAM) e `codecarbon` para estimativa de emissões de CO₂; se `codecarbon` falhar, o sistema entra em modo simulado.
- **Dispositivo quântico por defeito:** `default.qubit` (PennyLane). Para aceleração por GPU, altere o device nos ficheiros de teste individuais para `lightning.gpu` (requer PennyLane-Lightning).
- **Estratégias de treino:** técnicas para mitigar barren plateaus, inicialização "Cold Start" e schedulers manuais de learning rate para preservar sinal de gradiente.
- **Consumo computacional:** modelos como MPS (D) e TTN/MERA (F) podem exigir memória/tempo significativos; `BENCHMARK_MASTER.py` realiza uma auditoria inicial para adaptar execuções.
- **Formato de saída:** CSVs e imagens PNG para fácil visualização e inclusão em publicações; logs textuais para auditoria.

---

## 🧪 Testes e Verificação

- `benchmark_quantico_anti_bp.py` — "Arena Suprema": execução controlada dos modelos A–G com estratégias anti-barren plateau.
- `benchmark_classico_robusto.py` — Treino/avaliação dos modelos clássicos (Random Forest, SVM, Gradient Boosting, KNN).
- `teste_modelo_[A-G].py` — Scripts individuais para reproduzir cada arquitetura quântica.

---

## 📜 Licença e Autoria

Desenvolvido no âmbito do projeto de investigação **QPMF 2025**.

**Todos os direitos reservados.** 
Não remover créditos nem alterar a metodologia dos testes sem autorização prévia dos autores.

---

## 🤝 Contribuições

Contribuições são bem-vindas. 
Para alterações metodológicas importantes (arquitetura de pipeline, métricas de avaliação, manipulação de dados sintéticos), abra uma *issue* descrevendo o propósito e impacto.

--- Desenvolvido por Guilherme de Macedo Oliveira ---
