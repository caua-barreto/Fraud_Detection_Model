# 💳 Previsão e Análise de Risco de Crédito: IEEE-CIS Fraud Detection

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/c/ieee-fraud-detection)

Este projeto tem como objetivo desenvolver um sistema completo de ponta a ponta para a detecção de fraudes em transações de crédito, utilizando a base de dados da competição **IEEE-CIS Fraud Detection** do Kaggle. 

O foco deste projeto vai muito além de "treinar um modelo". O objetivo é construir uma solução **robusta, otimizada e altamente interpretável**, onde as métricas estatísticas estejam diretamente alinhadas ao valor de negócio, minimizando falsos positivos (bloqueios indevidos de clientes) e maximizando a interceptação de fraudadores reais.

---

## 📊 O Dataset

A base de dados utilizada é fornecida pela Vesta Corporation em parceria com a IEEE Computational Intelligence Society. Ela contém centenas de features anônimas e atributos categóricos simulando um ambiente real de e-commerce e pagamentos online. O desafio principal é lidar com um cenário extremo de **classe minoritária** (fraude representa cerca de 3.5% dos dados) e variáveis altamente dimensionais e correlacionadas.

🔗 **Link Oficial da Competição:** [IEEE-CIS Fraud Detection no Kaggle](https://www.kaggle.com/c/ieee-fraud-detection)

---

## 🗺️ Roadmap e Arquitetura do Projeto

O desenvolvimento deste repositório segue um fluxo de trabalho dividido em quatro grandes módulos, construídos de forma progressiva. *Abaixo está o detalhamento de cada etapa do que já foi feito ou está em desenvolvimento:*

### 1. `01_EDA.ipynb` (Análise Exploratória e Engenharia de Features)
Nesta fase, o foco é compreender a anatomia das fraudes e sanear os dados antes que toquem em qualquer algoritmo complexo.
* **Inspeção de Vínculos:** Entendimento profundo das features mascaradas (blocos C, D, M, V) e unificação correta das tabelas de transação (`transaction`) e identidade (`identity`).
* **Saneamento e Redução de Dimensionalidade:** Tratamento de valores nulos estruturais, normalização de caudas longas (outliers) e descarte cirúrgico de features com alta multicolinearidade e Information Value (IV) redundante para otimizar o uso de memória.
* **Inferência Estatística:** Criação de variáveis preditivas fortes (meta-features de frequência, cruzes de e-mail e deltas temporais) embasadas na lógica operacional de risco de crédito.

### 2. `02_Modelagem.ipynb` (Treinamento, Otimização e Regularização)
Aqui, o poder computacional entra em ação para lidar com o forte desbalanceamento das classes.
* **Algoritmos de Boosting Avançados:** Implementação de `LightGBM`, `XGBoost` e `CatBoost`, tirando proveito do tratamento nativo de dados esparsos.
* **Validação Robusta:** Substituição da validação aleatória por uma estratégia de **Time-Series Split (Cross Validation Temporal)** para evitar vazamento de dados (*data leakage*) e espelhar o ambiente de produção.
* **Tunning e Poda:** Busca Bayesiana de hiperparâmetros e poda final de variáveis não essenciais.
* **Tratamento de Classe Rara:** Em vez de simplesmente aplicar subamostragem (downsampling), o projeto aplica ajustes finos nos pesos das classes e utiliza regressão isotônica para garantir saídas de probabilidades calibradas e realistas.

### 3. `03_Avaliacao.ipynb` (Métricas de Decisão e Negócio)
Foco em avaliar o impacto real das previsões na base *Out-of-Time* (teste).
* **Além do Accuracy:** Acurácia é inútil em cenários de fraude. O modelo é avaliado usando **ROC-AUC, Precision-Recall (PR-AUC), e Índice de Gini**.
* **Calibração de Threshold:** Análise matemática da Matriz de Confusão para encontrar o ponto de corte ideal que balanceia a segurança da empresa e a fricção para o cliente legítimo.
* **Estabilidade e Concept Drift:** Monitoramento da degradação das métricas do modelo ao longo da passagem do tempo no conjunto de validação.

### 4. `04_Analise_de_Risco.ipynb` (Clusterização e *Modus Operandi*)
A última milha do projeto busca entregar clareza estratégica.
* **Clusterização de Fraudes:** Aplicação de redução de dimensionalidade e algoritmos de agrupamento na base dos classificados como fraude.
* **Descobrindo Padrões (SHAP):** Unindo clusterização aos valores SHAP para decifrar "como" os fraudadores operam em grupos distintos (ex: ataques de bots coordenados, roubos de conta isolados, testes massivos de cartão).
* **Insights Acionáveis:** Como os resultados desta análise de agrupamento podem realimentar e criar regras duras de bloqueio nos sistemas prévios de defesa da empresa.

---

## 📈 Noções de Crédito Aplicadas

Durante as etapas de modelagem e avaliação, adotamos termos e lógicas comuns a birôs de crédito e mesas de risco:
* **Information Value (IV) e Weight of Evidence (WoE):** Para ranquear o poder de separação das variáveis.
* **Falso Positivo como "Perda Comercial":** Em risco de fraude em pagamentos online, bloquear uma transação legítima por engano significa perder a venda, o lucro líquido e, frequentemente, o cliente. 
* **Falso Negativo como "Perda Financeira" (Chargeback):** Deixar a fraude passar gera estornos duplos e multas das bandeiras de cartão de crédito.
* **População de Inferência (Through the Door):** A base que passa pela porta da frente da loja, contendo a verdadeira proporção orgânica das transações.

