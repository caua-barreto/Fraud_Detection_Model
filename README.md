# 💳 Previsão e Análise de Risco de Crédito: IEEE-CIS Fraud Detection

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/c/ieee-fraud-detection)

Este projeto tem como objetivo desenvolver um sistema completo de ponta a ponta para a detecção de fraudes em transações de crédito, utilizando a base de dados da competição **IEEE-CIS Fraud Detection** do Kaggle.

---

### 📌 Status do Projeto (`12/11/2026`)
Pipeline principal estruturado de ponta a ponta (`01_EDA.ipynb`, `02_Modelagem.ipynb`, `04_Clusters.ipynb` e módulo central `utils.py`): **EDA e Master Table** concluídas, **Modelagem** com divisão temporal 60/20/20 e calibração/threshold financeiro ajustados, e **Clusterização** de perfis de fraude (`K-Means K=10`) mapeada em arquétipos de ataque. Próxima etapa focada em refinamentos analíticos e avaliação aprofundada no conjunto *Out-of-Time* (`df_oot`).

---

## 📊 O Dataset

A base de dados utilizada é fornecida pela Vesta Corporation em parceria com a IEEE Computational Intelligence Society. Ela contém centenas de features anônimas e atributos categóricos simulando um ambiente real de e-commerce e pagamentos online. O desafio principal é lidar com um cenário extremo de **classe minoritária** (fraude representa cerca de 3,5% dos dados) e variáveis altamente dimensionais e correlacionadas.

🔗 **Link Oficial da Competição:** [IEEE-CIS Fraud Detection no Kaggle](https://www.kaggle.com/c/ieee-fraud-detection)

---

## 🗺️ Roadmap e Arquitetura do Projeto

O desenvolvimento deste repositório segue um fluxo modular dividido em notebooks especializados e apoiado pelas funções de `utils.py`:

### 1. `notebooks/01_EDA.ipynb` (Análise Exploratória e Engenharia de Features)
* **Inspeção e Unificação:** Análise dos blocos transacionais e de identidade (`transaction` + `identity`), tratamento de nulos estruturais e otimização de memória (`safe_downcast`).
* **Engenharia de Atributos:** Criação de variáveis temporais, métricas financeiras (como `np.log1p` em caudas longas e razões por cartão), cruzes de domínio de e-mail, contagens de nulos e *frequency encoding*.
* **Funil de Seleção Multivariada:** Filtragem via **Information Value (IV)**, análise de correlação de pares e diagnóstico de **VIF**, consolidando a saída em `data/processed/df_master.parquet` e `features_metadata.json`.

### 2. `notebooks/02_Modelagem.ipynb` (Treinamento, Otimização e Calibração)
* **Split Temporal Estrito (60% / 20% / 20%):** Particionamento cronológico via `TransactionDT` em **Treino (`train_df`, 60%)**, **Validação (`df_validacao`, 20%)** e **Out-of-Time (`df_oot`, 20%)**, mantendo o OOT cego durante tuning, seleção de threshold e poda SHAP.
* **Cross-Validation sem Leakage:** Validação cruzada agrupada por mês (`GroupKFold`) exclusivamente no Treino, com agregações e encodings ajustados pós-split.
* **Algoritmos e Tuning:** Baselines (`LogisticRegression`, `LightGBM`, `XGBoost`, `CatBoost`), otimização Bayesiana com **Optuna**, checagem de correlação para ensemble e poda dimensional via **SHAP**.
* **Calibração e Threshold Financeiro:** Calibração isotônica em predições *Out-of-Fold* (OOF) e otimização do ponto de corte baseada em retorno líquido em US$ reais (`np.expm1` em `TransactionAmt`).

### 3. `notebooks/03_Av.ipynb` (Avaliação Avançada no OOT, Drift e Hipóteses)
* **Prova Final no OOT:** Avaliação estatística e operacional no bloco futuro (`df_oot`), monitorando **ROC-AUC, PR-AUC, KS, estabilidade temporal e Drift**.
* **Cruzamento com Clusters:** Análise do desempenho do modelo e validação de hipóteses de negócio segmentadas por perfil de ataque.

### 4. `notebooks/04_Clusters.ipynb` (Clusterização e *Modus Operandi* das Fraudes)
* **Escopo Livre de OOT:** Filtragem das transações fraudulentas em `Treino + Validação` mantendo o OOT isolado.
* **Redução Dimensional e Agrupamento:** Comparação entre **PCA** e **UMAP** (10D e 2D) e segmentação com **K-Means (`K=10`)**.
* **Mapeamento de Quadrilhas:** Perfilamento numérico via *heatmap* de Z-Score e consolidação de quatro macrogrupos estratégicos (Botnets/Card Testing, *Hit and Run* no Varejo, *Account Takeover* Premium e Fraude Clássica/Limbo Operacional).

---

## 🛠️ Mudanças Realizadas (Pré-Commit)

* **Consolidação da EDA (`notebooks/01_EDA.ipynb`):** Integração da etapa de engenharia de atributos diretamente na EDA (remoção de `notebooks/02_Feature_Engineering.ipynb`), cálculo de IV, remoção por correlação/VIF e exportação da `df_master.parquet`.
* **Implementação de `notebooks/02_Modelagem.ipynb`:**
  * Reestruturação completa para o split cronológico em 3 blocos (**60% Treino / 20% Validação / 20% OOT**), garantindo que o OOT não participe da busca de hiperparâmetros, comparação de modelos ou poda SHAP.
  * Ajuste da tabela de métricas (`Treino`, `Teste` no CV e `Validação` isolada) e verificação de correlação de Pearson/Spearman entre `LightGBM_Opt` e `XGBoost_Opt`.
  * Correção na Seção 6.0 (Calibração Isotônica e Threshold Financeiro) para reconverter `TransactionAmt` da escala `log1p` para dólares reais (`np.expm1`) e alinhar corretamente a avaliação com `df_validacao` (`X_test`/`y_test`), preservando o OOT cego até a Seção 8.0.
* **Implementação de `notebooks/04_Clusters.ipynb`:**
  * Estruturação da esteira de redução de dimensionalidade (`PCA` vs `UMAP`) e agrupamento (`K-Means K=10`).
  * Organização das células respeitando o *storytelling* analítico (no máximo um gráfico ou tabela por célula) e ajuste do código final com `plot_heatmap_zscore_clusters`.
* **Expansão de `utils.py` e `requirements.txt`:** Adição e refatoração das funções de preparação sem leakage (`preparar_dados_modelagem`), avaliação (`avaliar_classificador`, `avaliar_classificacao_oot`), funções objetivo do Optuna, poda SHAP, exportação de artefatos e gráficos/métricas de clusterização; criação do placeholder `notebooks/03_Av.ipynb`.

---

## ✅ Tarefas a Fazer (Próximos Passos)

1. **EDA (`notebooks/01_EDA.ipynb`):**
   - [ ] Adicionar comentários interpretativos ao longo das análises e decisões de saneamento.
   - [ ] Investigar e documentar possíveis incoerências nos dados (ex.: anomalias de preenchimento, valores extremos e consistência temporal).
   - [ ] Renomear as colunas importantes (especialmente variáveis anonimizadas dos blocos `C`, `D`, `M`, `V` e features derivadas) para facilitar a interpretabilidade de negócio.
2. **Modelagem (`notebooks/02_Modelagem.ipynb`):**
   - [ ] Finalizar a execução completa da modelagem após os ajustes finais do pipeline.
   - [ ] Concluir e validar a **Calibração Isotônica** e a curva de threshold financeiro em US$ reais.
   - [ ] Adicionar comentários explicativos detalhando trade-offs, escolha do modelo campeão e regras de negócio.
3. **Clusterização (`notebooks/04_Clusters.ipynb`):**
   - [ ] Justificar formalmente por que **não utilizar algoritmos baseados em densidade** (como DBSCAN/HDBSCAN) frente à geometria, densidade heterogênea e dimensionalidade das features de fraude.
   - [ ] Adicionar mais comentários interpretativos conectando os achados estatísticos dos clusters às ações antifraude.
4. **Avaliação no OOT (`notebooks/03_Avaliation.ipynb` / Seção OOT):**
   - [ ] Executar a avaliação final cega no conjunto **Out-of-Time (`df_oot`)**.
   - [ ] Analisar **estabilidade temporal, *data/concept drift*, KS, PSI** e evolução mensal das métricas.
   - [ ] Analisar o desempenho e a taxa de captura do modelo **sob a ótica dos clusters** identificados e **testar hipóteses** de negócio/risco.
