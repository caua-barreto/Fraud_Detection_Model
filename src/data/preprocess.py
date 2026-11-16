"""Tratamento de nulos estruturais e caudas longas (log1p) — detecção, aplicação e reversão.

Código migrado de `utils.py` (fachada de compatibilidade dos notebooks).
"""
from __future__ import annotations

from typing import Optional, Dict, List, Tuple, Iterable

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
from src.visualization.theme import _mostrar_figura


def tratar_nulos_id(df_id: pd.DataFrame) -> pd.DataFrame:
    """
    Trata o bloco esparso (>90% de nulos) de `df_train_id`:
    1. Cria meta-feature `id_null_count` com a contagem de nulos por linha.
    2. Sintetiza o bloco esparso (`id_07`, `id_08`, `id_21` a `id_27`) na flag `has_advanced_network_check`.
    3. Remove as colunas originais com >90% de nulos de forma idempotente.

    Parâmetros
    ----------
    df_id : pd.DataFrame
        DataFrame de identidade (`df_train_id`).

    Retorna
    -------
    pd.DataFrame
        DataFrame tratado.
    """
    cols_to_drop = ['id_07', 'id_08', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27']

    if 'id_null_count' not in df_id.columns:
        df_id['id_null_count'] = df_id.isnull().sum(axis=1).astype('int16')

    if 'id_23' in df_id.columns:
        df_id['has_advanced_network_check'] = df_id['id_23'].notnull().astype('int8')

    cols_existentes = [c for c in cols_to_drop if c in df_id.columns]
    if cols_existentes:
        df_id = df_id.drop(columns=cols_existentes)

    print(f"Colunas esparsas removidas ({len(cols_existentes)}). Colunas restantes em df_train_id: {df_id.shape[1]}")
    return df_id


def analisar_e_transformar_caudas_longas(
    df: pd.DataFrame,
    limiar_skew: float = 1.0,
    min_unicos: int = 10,
    substituir_original: bool = False,
    plotar_top: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Diagnostica assimetria (skewness) em variáveis numéricas contínuas/discretas,
    aplica np.log1p apenas onde é matematicamente seguro (min >= 0 e nunique > min_unicos)
    e compara o antes e depois.
    """
    df_out = df.copy()
    cols_ignorar = {'TransactionID', 'isFraud', 'TransactionDT'}
    
    # 1. Seleciona apenas colunas numéricas candidatas (exclui flags binárias e chaves)
    candidatas = []
    for col in df_out.select_dtypes(include=[np.number]).columns:
        if col in cols_ignorar or col.endswith('_num') or col.startswith(('has_', 'is_')):
            continue
        if df_out[col].nunique(dropna=True) > min_unicos:
            candidatas.append(col)

    # 2. Calcula assimetria e estatísticas de suporte
    skews = df_out[candidatas].skew(skipna=True)
    mins  = df_out[candidatas].min(skipna=True)
    maxs  = df_out[candidatas].max(skipna=True)

    relatorio = []
    cols_transformadas = []

    for col in candidatas:
        skew_val = skews[col]
        min_val  = mins[col]
        max_val  = maxs[col]

        if skew_val > limiar_skew:
            if min_val >= 0:
                serie_log = np.log1p(df_out[col]).astype('float32')
                skew_novo = float(serie_log.skew(skipna=True))
                
                target_col = col if substituir_original else f"{col}_log"
                df_out[target_col] = serie_log
                cols_transformadas.append((col, target_col, skew_val, skew_novo))
                status = "Transformada (log1p)"
            else:
                skew_novo = skew_val
                status = f"Ignorada (min={min_val:.1f} < 0)"
        else:
            skew_novo = skew_val
            status = "Simétrica / Aceitável"

        relatorio.append({
            "Feature": col,
            "Min": round(float(min_val), 2),
            "Max": round(float(max_val), 2),
            "Skew_Antes": round(float(skew_val), 2),
            "Skew_Depois": round(float(skew_novo), 2),
            "Status": status
        })

    df_relatorio = (
        pd.DataFrame(relatorio)
        .sort_values(by="Skew_Antes", ascending=False)
        .reset_index(drop=True)
    )

    print(f"Total de colunas avaliadas      : {len(candidatas)}")
    print(f"Colunas com Skew > {limiar_skew} (log1p): {len(cols_transformadas)}")
    print(f"Ignoradas por valores negativos : {(df_relatorio['Status'].str.startswith('Ignorada')).sum()}")

    # 3. Plot comparativo (Antes vs. Depois) das Top N features com maior redução de cauda
    if plotar_top > 0 and len(cols_transformadas) > 0:
        # Prioriza features de negócio conhecidas se estiverem na lista, senão pega as maiores
        destaques = [t for t in cols_transformadas if t[0] in ['TransactionAmt', 'dist1', 'C1', 'C13', 'D1']]
        if len(destaques) < plotar_top:
            restante = [t for t in cols_transformadas if t not in destaques]
            destaques.extend(restante[:plotar_top - len(destaques)])
        destaques = destaques[:plotar_top]

        fig, axes = plt.subplots(len(destaques), 2, figsize=(13, 3.5 * len(destaques)))
        if len(destaques) == 1:
            axes = np.array([axes])

        for idx, (col_orig, col_log, sk_old, sk_new) in enumerate(destaques):
            # Antes (usa df original caso substituir_original=True)
            dados_orig = df[col_orig].dropna()
            dados_log  = df_out[col_log].dropna()

            sns.histplot(dados_orig, bins=50, ax=axes[idx, 0], color='#48CAE4', kde=False)
            axes[idx, 0].set_title(f"{col_orig} — Original (Skew: {sk_old:.2f})")
            axes[idx, 0].set_yscale('log') # Escala log no eixo Y para enxergar a cauda

            sns.histplot(dados_log, bins=50, ax=axes[idx, 1], color='#0B2545', kde=True)
            axes[idx, 1].set_title(f"{col_log} — Pós log1p (Skew: {sk_new:.2f})")

        plt.tight_layout()
        _mostrar_figura()

    return df_out, df_relatorio


def _serie_parece_log1p(serie: pd.Series, limite_max: float, exigir_fracionario: bool) -> bool:
    """Heurística de escala `log1p`.

    - Valores monetários (`exigir_fracionario=False`): não-negativos e `max <= limite_max`
      (mesma regra de `gerar_perfil_fraude_clusters`, garantindo consistência entre tabelas).
    - Contagens (`exigir_fracionario=True`): além do limite, a maioria dos valores não-nulos
      precisa ser fracionária. Contagens reais são inteiras; `log1p(k)` só é inteiro em k=0.
    """
    v = pd.to_numeric(serie, errors="coerce").dropna().astype("float64")
    if v.empty or v.min() < 0 or v.max() > limite_max:
        return False
    if not exigir_fracionario:
        return True
    nao_zero = v[v != 0]
    if nao_zero.empty:
        return False
    frac_fracionarios = float((np.abs(nao_zero - np.round(nao_zero)) > 1e-6).mean())
    return frac_fracionarios > 0.5


def reverter_escala_log1p(
    df: pd.DataFrame,
    cols_monetarias: Iterable[str] = ("TransactionAmt",),
    cols_contagem: Optional[Iterable[str]] = None,
    limite_monetario: float = 20.0,
    limite_contagem: float = 15.0,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, bool]]:
    """Retorna cópia do DataFrame com colunas em `log1p` reconvertidas via `np.expm1`.

    A detecção é feita **na coluna inteira** (não por grupo), evitando reconversões
    inconsistentes entre clusters. Retorna também `{coluna: foi_reconvertida}`.
    """
    df_out = df.copy()
    if cols_contagem is None:
        cols_contagem = [f"C{i}" for i in range(1, 15)] + ["tr_null_count", "id_null_count"]

    relatorio: Dict[str, bool] = {}
    for col, limite, exige_frac in (
        [(c, limite_monetario, False) for c in cols_monetarias]
        + [(c, limite_contagem, True) for c in cols_contagem]
    ):
        if col not in df_out.columns or not pd.api.types.is_numeric_dtype(df_out[col]):
            continue
        em_log = _serie_parece_log1p(df_out[col], limite, exige_frac)
        if em_log:
            df_out[col] = np.expm1(df_out[col].astype("float64"))
        else:
            df_out[col] = df_out[col].astype("float64")
        relatorio[col] = em_log

    if verbose:
        convertidas = [c for c, ok in relatorio.items() if ok]
        mantidas = [c for c, ok in relatorio.items() if not ok]
        print(f"Escala real (expm1) aplicada em : {convertidas if convertidas else 'nenhuma'}")
        print(f"Já em escala real (mantidas)    : {mantidas if mantidas else 'nenhuma'}")
    return df_out, relatorio


def _detectar_e_aplicar_log1p_referencia(
    df_alvo: pd.DataFrame,
    df_ref: pd.DataFrame,
    cols_log_explicitas: Optional[List[str]] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Aplica `np.log1p` nas colunas de `df_alvo` cuja contraparte em `df_ref` (df_master)
    foi transformada por `analisar_e_transformar_caudas_longas` no 01_EDA.

    Usa a lista explícita de metadados quando fornecida; caso contrário, decide pela menor
    distância KS após igualar a precisão `float32` e arredondar para 4 casas decimais
    (evitando falsos degraus de precisão `float32` vs. `float64` em variáveis discretas como `vel_C2`).
    """
    df_out = df_alvo.copy()
    cols_ignorar = {'TransactionID', 'isFraud', 'TransactionDT'}
    cols_log: List[str] = []

    if cols_log_explicitas:
        for col in cols_log_explicitas:
            if col in df_out.columns and col not in cols_ignorar:
                df_out[col] = np.log1p(df_out[col].astype('float64').clip(lower=0)).astype('float32')
                cols_log.append(col)
        return df_out, cols_log

    for col in df_out.select_dtypes(include=[np.number]).columns:
        if col in cols_ignorar or col.endswith('_num') or col.startswith(('has_', 'is_')) or col not in df_ref.columns:
            continue
        m = df_ref[col].dropna()
        t = df_out[col].dropna()
        if len(m) == 0 or len(t) == 0 or m.nunique() <= 10 or m.min() < 0:
            continue
        m_s = np.round(m.sample(min(len(m), 40_000), random_state=random_state).astype('float32').astype('float64'), 4)
        t_raw = t.sample(min(len(t), 40_000), random_state=random_state).astype('float64')
        t_s = np.round(t_raw.astype('float32').astype('float64'), 4)
        t_log = np.round(np.log1p(t_raw.clip(lower=0)).astype('float32').astype('float64'), 4)
        ks_raw = ks_2samp(m_s, t_s).statistic
        ks_log = ks_2samp(m_s, t_log).statistic
        if ks_log < ks_raw:
            df_out[col] = np.log1p(df_out[col].astype('float64').clip(lower=0)).astype('float32')
            cols_log.append(col)

    return df_out, cols_log
