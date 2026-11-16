"""Seleção de features: IV, correlação de pares, VIF, multicolinearidade e poda por SHAP.

Código migrado de `utils.py` (fachada de compatibilidade dos notebooks).
"""
from __future__ import annotations

from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from src.visualization.theme import _mostrar_figura

try:
    from IPython.display import display
except ImportError:  # pragma: no cover
    display = print
try:
    import shap
except ImportError:  # pragma: no cover
    shap = None
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant
except ImportError:  # pragma: no cover
    variance_inflation_factor = None
    add_constant = None


def analise_correlacao_pares(
    df: pd.DataFrame,
    target_col: str = 'isFraud',
    limite: float = 0.70,
    threshold: Optional[float] = None
) -> pd.DataFrame:
    """
    Identifica pares de features numéricas com correlação linear absoluta acima do limite.
    Calcula dinamicamente o Information Value (IV) em relação ao target e imprime
    a feature de maior poder preditivo sempre à esquerda (candidata a manter).

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame contendo as features e o target.
    target_col : str, default='isFraud'
        Nome da coluna target para cálculo do IV de desempate.
    limite : float, default=0.70
        Limiar de correlação absoluta.
    threshold : float, optional
        Alias para `limite` (caso passado como `threshold=0.7`).

    Retorna
    -------
    pd.DataFrame
        Tabela com os pares colineares ordenados pela correlação decrescente.
    """
    if threshold is not None:
        limite = threshold

    def calcular_iv_interno(feature: str) -> float:
        if target_col not in df.columns:
            return 0.0
        temp = df[[feature, target_col]].dropna()
        if temp.empty:
            return 0.0

        if temp[feature].nunique() <= 10:
            binned = temp[feature]
        else:
            try:
                binned = pd.qcut(temp[feature], q=10, duplicates='drop')
            except Exception:
                return 0.0

        agrupado = temp.groupby(binned, observed=False)[target_col].agg(['count', 'sum'])
        agrupado['eventos'] = agrupado['sum']
        agrupado['nao_eventos'] = agrupado['count'] - agrupado['sum']

        total_eventos = agrupado['eventos'].sum()
        total_nao_eventos = agrupado['nao_eventos'].sum()

        if total_eventos == 0 or total_nao_eventos == 0:
            return 0.0

        taxa_eventos = (agrupado['eventos'] / total_eventos).replace(0, 0.0001)
        taxa_nao_eventos = (agrupado['nao_eventos'] / total_nao_eventos).replace(0, 0.0001)

        woe = np.log(taxa_nao_eventos / taxa_eventos)
        iv = (taxa_nao_eventos - taxa_eventos) * woe
        return float(iv.sum())

    cols_ignorar = [target_col, 'TransactionID', 'case_id']
    cols_numericas = df.drop(columns=cols_ignorar, errors='ignore').select_dtypes(include=[np.number])
    matriz_corr = cols_numericas.corr().abs()

    triangulo_sup = matriz_corr.where(np.triu(np.ones(matriz_corr.shape), k=1).astype(bool))

    pares_altos = [
        (linha, coluna, triangulo_sup.loc[linha, coluna])
        for coluna in triangulo_sup.columns
        for linha in triangulo_sup.index
        if triangulo_sup.loc[linha, coluna] > limite
    ]

    features_colineares = set([p[0] for p in pares_altos] + [p[1] for p in pares_altos])
    dict_iv_dinamico = {f: calcular_iv_interno(f) for f in features_colineares}

    print(f"--- Pares com Correlação > {limite} (Total: {len(pares_altos)} pares) ---")

    registros = []
    for par in sorted(pares_altos, key=lambda x: x[2], reverse=True):
        feat1, feat2, corr = par[0], par[1], par[2]
        iv1 = dict_iv_dinamico.get(feat1, 0.0)
        iv2 = dict_iv_dinamico.get(feat2, 0.0)

        if iv1 >= iv2:
            feat_manter, feat_descartar = feat1, feat2
            iv_manter, iv_descartar = iv1, iv2
        else:
            feat_manter, feat_descartar = feat2, feat1
            iv_manter, iv_descartar = iv2, iv1

        print(f"{feat_manter} (IV: {iv_manter:.4f}) & {feat_descartar} (IV: {iv_descartar:.4f}) | Corr: {corr:.3f}")
        registros.append({
            'Feature_Manter': feat_manter,
            'IV_Manter': round(iv_manter, 4),
            'Feature_Descartar': feat_descartar,
            'IV_Descartar': round(iv_descartar, 4),
            'Correlacao': round(float(corr), 4)
        })

    return pd.DataFrame(registros)


# Aliases para compatibilidade
analise_correlação_pares = analise_correlacao_pares


analisar_multicolinearidade_pares = analise_correlacao_pares


analisar_multicolinearidade = analise_correlacao_pares


def calcular_iv_dataset(
    df: pd.DataFrame,
    target_col: str = 'isFraud',
    bins: int = 10
) -> pd.DataFrame:
    """
    Calcula o Information Value (IV) para todas as features preditoras do DataFrame,
    com a régua de diagnóstico adaptada para Detecção de Fraude Transacional.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame contendo as features e a coluna target.
    target_col : str, default='isFraud'
        Nome da variável alvo binária (0/1).
    bins : int, default=10
        Quantidade de decis para discretizar variáveis numéricas contínuas.

    Retorna
    -------
    pd.DataFrame
        Tabela ordenada por `Information_Value` decrescente com o diagnóstico ajustado.
    """
    iv_resultados = []

    mask_valid = df[target_col].notna()
    y = df.loc[mask_valid, target_col]
    total_bons = int((y == 0).sum())
    total_maus = int((y == 1).sum())
    eps = 1e-6

    colunas_ignorar = {'TransactionID', 'case_id', 'date_decision', 'MONTH', 'WEEK_NUM', target_col}
    features = [c for c in df.columns if c not in colunas_ignorar]

    for col in features:
        serie = df.loc[mask_valid, col]

        if pd.api.types.is_numeric_dtype(serie):
            try:
                categorias = pd.qcut(serie, q=bins, duplicates='drop').astype(str)
            except Exception:
                categorias = pd.cut(serie, bins=bins).astype(str)
            categorias = categorias.replace('nan', 'Missing').fillna('Missing')
        else:
            categorias = serie.astype(str).replace('nan', 'Missing').fillna('Missing')

        tabela = y.groupby(categorias, observed=False).agg(
            total='count',
            maus='sum'
        ).reset_index()
        tabela['bons'] = tabela['total'] - tabela['maus']

        tabela['dist_bons'] = (tabela['bons'] + eps) / total_bons
        tabela['dist_maus'] = (tabela['maus'] + eps) / total_maus

        tabela['woe'] = np.log(tabela['dist_bons'] / tabela['dist_maus'])
        iv_total = float(((tabela['dist_bons'] - tabela['dist_maus']) * tabela['woe']).sum())

        # ---------------------------------------------------------
        # RÉGUA DE DIAGNÓSTICO ADAPTADA PARA FRAUDE TRANSACIONAL
        # ---------------------------------------------------------
        if iv_total < 0.02:
            status = 'Sem Poder Preditivo'
        elif iv_total < 0.10:
            status = 'Poder Fraco'
        elif iv_total < 0.30:
            status = 'Poder Médio'
        elif iv_total < 0.80:
            status = 'Poder Forte'
        else:
            # Substitui o "Leakage" de Siddiqi por um reconhecimento de sinais extremos de fraude
            status = 'Poder Altíssimo (Padrão Fraude)'

        iv_resultados.append({
            'Feature': col,
            'Information_Value': round(iv_total, 4),
            'Diagnóstico': status
        })

    df_iv = pd.DataFrame(iv_resultados).sort_values(by='Information_Value', ascending=False).reset_index(drop=True)
    return df_iv


def analisar_vif(
    df: pd.DataFrame,
    limite: float = 10.0,
    imputar_nulos: bool = True,
    sample_size: Optional[int] = 100_000,
    random_state: int = 42,
    target_col: str = 'isFraud'
) -> pd.DataFrame:
    """
    Calcula o Fator de Inflação da Variância (VIF) para diagnosticar
    multicolinearidade multivariada entre as variáveis numéricas.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame contendo as variáveis preditoras.
    limite : float, default=10.0
        Valor de corte para considerar a multicolinearidade alta (VIF > 10).
    imputar_nulos : bool, default=True
        Se True, imputa NaNs com a mediana antes do cálculo.
    sample_size : int, optional, default=100_000
        Tamanho da subamostra aleatória de linhas para acelerar o cálculo em bases grandes.
        Se None, utiliza todas as linhas do DataFrame.
    random_state : int, default=42
        Semente para reprodutibilidade da amostragem.
    target_col : str, default='isFraud'
        Nome da variável alvo a ser excluída do cálculo de VIF.

    Retorna
    -------
    pd.DataFrame
        Tabela ordenada com todas as features e seus respectivos valores de VIF.
    """
    cols_ignorar = [target_col, 'TransactionID', 'case_id']
    df_proc = df.drop(columns=cols_ignorar, errors='ignore')

    if sample_size is not None and len(df_proc) > sample_size:
        print(f"Amostrando {sample_size:,} de {len(df_proc):,} linhas para cálculo do VIF (random_state={random_state})...")
        df_proc = df_proc.sample(n=sample_size, random_state=random_state)

    df_num = df_proc.select_dtypes(include=[np.number]).astype('float64')

    # Remove colunas totalmente nulas ou constantes na amostra
    cols_validas = [c for c in df_num.columns if df_num[c].notna().any() and df_num[c].nunique(dropna=True) > 1]
    df_num = df_num[cols_validas]

    if imputar_nulos:
        df_num = df_num.fillna(df_num.median())
    else:
        df_num = df_num.dropna()
        if df_num.empty:
            raise ValueError("O DataFrame ficou vazio após remover os NaNs. Use imputar_nulos=True.")

    X = add_constant(df_num)
    colunas = X.columns
    valores = X.values

    vif_resultados = []
    for i in range(len(colunas)):
        if colunas[i] == 'const':
            continue
        try:
            vif_val = variance_inflation_factor(valores, i)
        except Exception:
            vif_val = np.inf

        vif_resultados.append({
            'Feature': colunas[i],
            'VIF': float(vif_val)
        })

    df_vif = pd.DataFrame(vif_resultados).sort_values(by='VIF', ascending=False).reset_index(drop=True)

    altos = df_vif[df_vif['VIF'] > limite]
    print(f"--- Variáveis com VIF > {limite} (Total: {len(altos)} de {len(df_vif)} variáveis numéricas) ---")
    if altos.empty:
        print("Nenhuma variável ultrapassou o limite de VIF estipulado.")
    else:
        for _, row in altos.head(30).iterrows():
            print(f"  {row['Feature']}: {row['VIF']:.2f}")
        if len(altos) > 30:
            print(f"  ... e mais {len(altos) - 30} variáveis com VIF > {limite}.")

    return df_vif


def prune_multicollinearity(
    df: pd.DataFrame,
    cols_remover: List[str],
) -> pd.DataFrame:
    """
    Remove colunas com multicolinearidade severa (lista validada).
    """
    cols_existentes = [c for c in cols_remover if c in df.columns]
    if cols_existentes:
        df = df.drop(columns=cols_existentes)
        print(f"Multicolinearidade: {len(cols_existentes)} coluna(s) removida(s):")
        for c in cols_existentes:
            print(f"  - {c}")
    else:
        print("Multicolinearidade: nenhuma coluna da lista encontrada para remoção.")
    return df


def obter_menores_features(
    modelo,
    X_amostra: pd.DataFrame,
    bottom_n: int = 20,
    exibir: bool = True,
) -> pd.DataFrame:
    """
    Calcula a média absoluta dos SHAP values (|SHAP|) para cada variável em uma amostra
    e retorna/exibe as variáveis com menor impacto ('fraqueza' preditiva).
    """
    try:
        import shap
    except ImportError:
        print("[AVISO] A biblioteca 'shap' não está instalada. Execute: pip install shap")
        return pd.DataFrame()

    try:
        from IPython.display import display
    except ImportError:
        display = print

    clf = modelo
    if hasattr(modelo, 'named_steps'):
        if 'clf' in modelo.named_steps:
            clf = modelo.named_steps['clf']
        elif 'classifier' in modelo.named_steps:
            clf = modelo.named_steps['classifier']
        elif 'model' in modelo.named_steps:
            clf = modelo.named_steps['model']
        else:
            clf = modelo[-1]

    try:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_amostra)
    except Exception:
        explainer = shap.Explainer(clf, X_amostra)
        shap_values = explainer(X_amostra).values

    if isinstance(shap_values, list):
        shap_matriz = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_matriz = shap_values[:, :, 1]
    else:
        shap_matriz = shap_values

    shap_media_abs = np.abs(shap_matriz).mean(axis=0)

    df_bottom = (
        pd.DataFrame({
            'Feature': list(X_amostra.columns),
            'fraqueza': shap_media_abs
        })
        .sort_values('fraqueza', ascending=True)
        .head(bottom_n)
        .reset_index(drop=True)
    )

    if exibir:
        display(df_bottom)

    return df_bottom


def podar_features_por_shap(dicionario_modelos, X_train, y_train, X_test, y_test, X_amostra, top_n_remover=8):
    """
    Para cada modelo de árvore:
      1. calcula o impacto global |SHAP| médio via TreeExplainer em X_amostra;
      2. identifica as `top_n_remover` features de menor impacto;
      3. re-treina uma versão LITE (mesmos hiperparâmetros) sem essas features;
      4. compara o ROC-AUC original vs LITE em X_test (aceita se delta >= -0.0010).
    """
    try:
        import shap
    except ImportError:
        print("[AVISO] A biblioteca 'shap' não está instalada. Execute: pip install shap")
        return {}

    from sklearn.metrics import roc_auc_score

    features_sugeridas_para_corte = {}

    for nome, modelo in dicionario_modelos.items():
        if 'Pipeline' in str(type(modelo)) or 'Logistic' in nome:
            continue

        print(f'\n{"=" * 65}')
        print(f' ANALISE SHAP E SANITY CHECK DE PODA: {nome} ')
        print(f'{"=" * 65}')

        try:
            explainer = shap.TreeExplainer(modelo)
            shap_values = explainer.shap_values(X_amostra)
        except Exception:
            explainer = shap.Explainer(modelo, X_amostra)
            shap_values = explainer(X_amostra).values

        if isinstance(shap_values, list):
            shap_values_positivos = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values_positivos = shap_values[:, :, 1]
        else:
            shap_values_positivos = shap_values

        importancia_global = np.abs(shap_values_positivos).mean(axis=0)
        df_importancia = pd.DataFrame({
            'Feature': list(X_amostra.columns),
            'SHAP_Impact': importancia_global
        }).sort_values(by='SHAP_Impact', ascending=True)

        piores_features = df_importancia.head(top_n_remover)['Feature'].tolist()

        print(f'As {top_n_remover} features menos relevantes:')
        for f in piores_features:
            print(f'  - {f}')

        probs_originais = modelo.predict_proba(X_test)[:, 1]
        auc_original = roc_auc_score(np.asarray(y_test), probs_originais)

        X_train_podado = X_train.drop(columns=piores_features)
        X_test_podado  = X_test.drop(columns=piores_features)

        modelo_podado = modelo.__class__(**modelo.get_params())
        print(f'\nTreinando arquitetura LITE sem {top_n_remover} features...')
        if 'CatBoost' in nome or 'XGB' in nome:
            modelo_podado.fit(X_train_podado, y_train, verbose=False)
        else:
            modelo_podado.fit(X_train_podado, y_train)

        probs_podadas = modelo_podado.predict_proba(X_test_podado)[:, 1]
        auc_podado = roc_auc_score(np.asarray(y_test), probs_podadas)
        delta_auc = auc_podado - auc_original

        print(f'  ROC-AUC Original : {auc_original:.4f}')
        print(f'  ROC-AUC Lite     : {auc_podado:.4f}')
        print(f'  Delta (AUC)      : {delta_auc:+.4f}')

        if delta_auc >= -0.0010:
            print('\n  [VEREDITO] Valido! Modelo mais limpo e rapido sem perda significativa.')
            features_sugeridas_para_corte[nome] = piores_features
        else:
            print('\n  [VEREDITO] Rejeitado! Features tinham interacoes ocultas.')
            features_sugeridas_para_corte[nome] = []

    return features_sugeridas_para_corte


def curva_poda_features_shap(
    modelo,
    ranking_features: List[str],
    X_train: pd.DataFrame,
    y_train,
    X_test: pd.DataFrame,
    y_test,
    metrica: str = 'ROC_AUC',  # Opções: 'ROC_AUC' ou 'GINI'
    passo: int = 10,
    min_features: int = 10,
    tolerancia: float = 0.0010,
    figsize: tuple = (12, 5),
    titulo: str = None,
    cor: str = '#13579B',
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Re-treina o modelo removendo features em lotes baseados no ranking SHAP.
    Avalia os resultados sem usar prints, exportando tudo em DataFrames.
    
    Retorna:
    - df_curva: Histórico completo de todas as iterações.
    - df_resumo: Tabela comparativa dos 2 modelos alvos.
    - features_max: Lista de features do modelo de maior índice absoluto.
    - features_otimizadas: Lista de features do modelo mais enxuto dentro da tolerância.
    """
    y_te = np.asarray(y_test)
    total = len(ranking_features)
    grid_n = list(range(total, min_features - 1, -passo))
    
    if grid_n[-1] != min_features and min_features < total:
        grid_n.append(min_features)

    nome_cls = modelo.__class__.__name__
    linhas = []
    
    # 1. Loop de treinamento sem prints
    for n in grid_n:
        cols = ranking_features[:n]
        m = modelo.__class__(**modelo.get_params())
        
        if 'CatBoost' in nome_cls or 'XGB' in nome_cls:
            m.fit(X_train[cols], y_train, verbose=False)
        else:
            m.fit(X_train[cols], y_train)
            
        probs = m.predict_proba(X_test[cols])[:, 1]
        
        roc = roc_auc_score(y_te, probs)
        gini = (2 * roc) - 1
        pr = average_precision_score(y_te, probs)
        
        linhas.append({
            'N_Features': n,
            'Removidas': total - n,
            'ROC_AUC': roc,
            'GINI': gini,
            'PR_AUC': pr,
        })

    df_curva = pd.DataFrame(linhas)
    col_metrica = metrica.upper()

    # 2. Definição dos 2 modelos
    val_full = float(df_curva.loc[df_curva['N_Features'] == total, col_metrica].iloc[0])

    # Modelo 1: Maior Índice Absoluto
    idx_max = df_curva[col_metrica].idxmax()
    n_max = int(df_curva.loc[idx_max, 'N_Features'])
    val_max = float(df_curva.loc[idx_max, col_metrica])

    # Modelo 2: Mais Otimizado na faixa de tolerância
    aceitaveis = df_curva[df_curva[col_metrica] >= (val_full - tolerancia)]
    n_otimizado = int(aceitaveis['N_Features'].min())
    val_otimizado = float(df_curva.loc[df_curva['N_Features'] == n_otimizado, col_metrica].iloc[0])

    # 3. DataFrame de Resumo (Substitui os antigos prints)
    df_resumo = pd.DataFrame([
        {
            'Perfil_Modelo': '1. Maior Índice Absoluto',
            'N_Features': n_max,
            'Removidas': total - n_max,
            col_metrica: val_max,
            'Delta_vs_Completo': val_max - val_full
        },
        {
            'Perfil_Modelo': '2. Otimizado (Lim. Tolerância)',
            'N_Features': n_otimizado,
            'Removidas': total - n_otimizado,
            col_metrica: val_otimizado,
            'Delta_vs_Completo': val_otimizado - val_full
        }
    ])

    # 4. Construção do Gráfico
    if titulo is None:
        titulo = f'{col_metrica} em Função da Redução de Features (ranking |SHAP|)'

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df_curva['N_Features'], df_curva[col_metrica], color=cor, lw=2.2, marker='o', ms=5, label=f'{col_metrica} (Validação)')
    ax.axhline(val_full, color='#0B2545', ls=':', lw=1.2, label=f'Completo ({total} feat) = {val_full:.4f}')
    ax.axhspan(val_full - tolerancia, val_full + tolerancia, color='#8DB6E3', alpha=0.22, label=f'Tolerância ± {tolerancia:.4f}')
    
    # Marcador do modelo otimizado
    ax.axvline(n_otimizado, color='#0096C7', ls='--', lw=1.8, label=f'Otimizado: {n_otimizado} feat ({val_otimizado:.4f})')
    
    # Marcador do modelo de maior índice (se diferir do otimizado)
    if n_max != n_otimizado:
        ax.axvline(n_max, color=cor, ls='-.', lw=1.2, alpha=0.8, label=f'Maior Índice: {n_max} feat ({val_max:.4f})')
    else:
        ax.axhline(val_max, color=cor, ls='-.', lw=1.2, alpha=0.8, label=f'Maior {col_metrica} = {val_max:.4f}')

    ax.invert_xaxis()
    ax.set_title(titulo, fontsize=12, fontweight='bold', loc='left')
    ax.set_xlabel('Número de Features Mantidas (menos relevantes removidas primeiro)')
    ax.set_ylabel(f'{col_metrica} (Validação)')
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    
    # Legenda fora da curva para não misturar visualmente
    ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))

    # Limpar bordas
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    _mostrar_figura()

    # 5. Listas de Features exportadas
    features_max = list(ranking_features[:n_max])
    features_otimizadas = list(ranking_features[:n_otimizado])

    return df_curva, df_resumo, features_max, features_otimizadas


# ==============================================================================
# PIPELINE REPRODUTÍVEL (scripts/01_prepare_data.py): FUNIL DO df_master (01_EDA 7.2–7.4)
# ==============================================================================
COLS_PROTEGIDAS = ['TransactionID', 'isFraud', 'TransactionDT']
COLS_BRUTAS_SUBSTITUIDAS = [
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
    'card6', 'id_12', 'id_15', 'id_16', 'id_28', 'id_29',
    'id_31', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38',
    'DeviceType', 'DeviceInfo', 'flag_V14_zero', 'V14',
]
COLS_FREQ_GLOBAIS = ['card1', 'card2', 'addr1', 'P_emaildomain']


def funil_selecao_df_master(
    df_train: pd.DataFrame,
    df_iv: pd.DataFrame,
    iv_min: float = 0.02,
    corr_max: float = 0.75,
    tolerancia_nulos: float = 0.20,
    linear_top_n: int = 50,
    linear_max_nulos: float = 0.30,
    downcast: bool = True,
) -> Tuple[pd.DataFrame, dict, dict]:
    """
    Reproduz o funil de saneamento do 01_EDA (Seção 7.2.1 a 7.4):
      1. descarta colunas textuais brutas já convertidas (_bin/_ord/_cat/_score);
      1.1 remove as `freq_*` globais (leakage temporal — recalculadas pós-split no 02);
      2. remove ruído (IV < `iv_min`);
      3. poda correlação |r| > `corr_max` protegendo o maior IV, com trava de nulos;
      4. downcast + remoção da singularidade V240/V241 (maior IV mantido);
      5. metadados (numéricas, categóricas e top-N densas para o baseline linear).

    Retorna `(df_master, metadata, resumo)`.
    """
    from src.data.loader import safe_downcast

    iv_dict = dict(zip(df_iv['Feature'], df_iv['Information_Value']))
    for new_c in ['valor_razao_media_card1'] + [f'freq_{c}' for c in COLS_FREQ_GLOBAIS]:
        if new_c in df_train.columns and new_c not in iv_dict:
            iv_dict[new_c] = 0.25

    cols_drop_duplicadas = [c for c in COLS_BRUTAS_SUBSTITUIDAS if c in df_train.columns]
    cols_drop_leakage_freq = [f'freq_{c}' for c in COLS_FREQ_GLOBAIS if f'freq_{c}' in df_train.columns]
    cols_drop_baixo_iv = [
        col for col, iv in iv_dict.items()
        if iv < iv_min and col in df_train.columns and col not in COLS_PROTEGIDAS
    ]
    # mesma ordem de remoção do notebook (set -> list) — a ordem não afeta o resultado
    df_temp = df_train.drop(columns=list(set(cols_drop_duplicadas + cols_drop_baixo_iv + cols_drop_leakage_freq)))

    num_cols = [c for c in df_temp.select_dtypes(include=[np.number]).columns if c not in COLS_PROTEGIDAS]
    num_cols_ordenadas = sorted(num_cols, key=lambda x: iv_dict.get(x, 0.0), reverse=True)
    corr_matrix = df_temp[num_cols_ordenadas].corr().abs()
    null_series = df_temp[num_cols_ordenadas].isnull().mean()

    cols_drop_corr = set()
    for i, col_forte in enumerate(num_cols_ordenadas):
        if col_forte in cols_drop_corr:
            continue
        for col_fraca in num_cols_ordenadas[i + 1:]:
            if col_fraca in cols_drop_corr:
                continue
            if corr_matrix.loc[col_forte, col_fraca] > corr_max:
                if null_series[col_forte] <= null_series[col_fraca] + tolerancia_nulos:
                    cols_drop_corr.add(col_fraca)

    df_master = df_temp.drop(columns=list(cols_drop_corr))
    if downcast:
        df_master = safe_downcast(df_master)

    col_drop_vif = None
    if 'V240' in df_master.columns and 'V241' in df_master.columns:
        col_drop_vif = 'V241' if iv_dict.get('V240', 0) >= iv_dict.get('V241', 0) else 'V240'
        df_master = df_master.drop(columns=[col_drop_vif])

    cat_features = [c for c in df_master.select_dtypes(include=['object', 'category']).columns
                    if c not in COLS_PROTEGIDAS]
    num_features = [c for c in df_master.select_dtypes(include=[np.number]).columns if c not in COLS_PROTEGIDAS]
    cols_excluir_linear = {'seg_cont_T_id35_38'}
    linear_features = [
        c for c in sorted(num_features, key=lambda x: iv_dict.get(x, 0), reverse=True)
        if df_master[c].isnull().mean() < linear_max_nulos and c not in cols_excluir_linear
    ][:linear_top_n]

    metadata = {
        'all_features': num_features + cat_features,
        'num_features': num_features,
        'cat_features': cat_features,
        'linear_baseline_features': linear_features,
    }
    resumo = {
        'shape_df_train': list(df_train.shape),
        'shape_df_master': list(df_master.shape),
        'removidas_brutas': len(cols_drop_duplicadas),
        'removidas_baixo_iv': len(cols_drop_baixo_iv),
        'removidas_freq_leakage': cols_drop_leakage_freq,
        'removidas_correlacao': len(cols_drop_corr),
        'removida_singular_vif': col_drop_vif,
        'parametros': {'iv_min': iv_min, 'corr_max': corr_max, 'tolerancia_nulos': tolerancia_nulos},
    }
    print("=" * 65)
    print(f"FUNIL df_master: {tuple(df_train.shape)} -> {tuple(df_master.shape)} | "
          f"brutas={len(cols_drop_duplicadas)} | IV<{iv_min}={len(cols_drop_baixo_iv)} | "
          f"corr>{corr_max}={len(cols_drop_corr)} | singular={col_drop_vif}")
    print(f" -> Numéricas: {len(num_features)} | Categóricas: {len(cat_features)} | Linear: {len(linear_features)}")
    print("=" * 65)
    return df_master, metadata, resumo
