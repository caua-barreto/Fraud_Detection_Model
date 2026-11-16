"""Carga/otimização de memória e split temporal estrito 60/20/20 (com engenharia pós-split).

Código migrado de `utils.py` (fachada de compatibilidade dos notebooks).
"""
from __future__ import annotations

from typing import Optional, List

import numpy as np
import pandas as pd


def safe_downcast(
    df: pd.DataFrame,
    ignorar_colunas: Optional[List[str]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Reduz tipos `float64` para `float32` e `int64` para `int8`/`int16`/`int32`
    sem perda de precisão analítica, preservando a chave primária (`TransactionID`).

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame a ser otimizado.
    ignorar_colunas : list[str], optional
        Colunas que não devem sofrer downcast (default=['TransactionID']).
    verbose : bool, default=True
        Se True, exibe a economia de memória obtida.

    Retorna
    -------
    pd.DataFrame
        DataFrame com tipos otimizados.
    """
    if ignorar_colunas is None:
        ignorar_colunas = ['TransactionID']

    mem_antes = df.memory_usage().sum() / (1024 ** 2)

    for col in df.columns:
        if col in ignorar_colunas:
            continue
        col_type = df[col].dtype
        if col_type == 'float64':
            df[col] = df[col].astype('float32')
        elif col_type == 'int64':
            c_min, c_max = df[col].min(), df[col].max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype('int8')
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype('int16')
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype('int32')

    mem_depois = df.memory_usage().sum() / (1024 ** 2)
    if verbose:
        reducao_pct = ((mem_antes - mem_depois) / mem_antes) * 100 if mem_antes > 0 else 0.0
        print(f"Downcast concluído: {mem_antes:.1f} MB -> {mem_depois:.1f} MB (-{reducao_pct:.1f}%)")

    return df


def preparar_dados_modelagem(
    df_master: pd.DataFrame,
    meta: Optional[dict] = None,
    train_ratio: float = 0.60,
    test_ratio: float = 0.20,
) -> dict:
    """
    Realiza a divisão temporal estrita em 3 blocos cronológicos:
    - Treino (`train_ratio`, padrão 60% iniciais)
    - Teste de validação (`test_ratio`, padrão 20% seguintes)
    - Out-of-Time final (`1 - train_ratio - test_ratio`, padrão 20% finais cegos)

    Aplica engenharia de grupo e frequency encoding
    fitados EXCLUSIVAMENTE no Treino (`train_df`), ordinal encoding sem leakage, cálculo
    da taxa de desbalanceamento e blocos mensais (`grupos_mes`) apenas no Treino.
    """
    from sklearn.preprocessing import OrdinalEncoder

    if train_ratio <= 0 or test_ratio <= 0 or (train_ratio + test_ratio) >= 1.0:
        raise ValueError("`train_ratio` e `test_ratio` devem ser > 0 e somar menos que 1.0 para preservar o bloco OOT.")

    meta = meta or {}

    # 1. Ordenação estritamente cronológica em 3 blocos: Treino (60%) | Teste (20%) | OOT (20%)
    df_sorted = df_master.sort_values("TransactionDT").reset_index(drop=True)
    n_total = len(df_sorted)
    idx_train = int(n_total * train_ratio)
    idx_test = int(n_total * (train_ratio + test_ratio))

    train_df = df_sorted.iloc[:idx_train].copy()
    test_df  = df_sorted.iloc[idx_train:idx_test].copy()
    oot_df   = df_sorted.iloc[idx_test:].copy()

    # 2. (Sem ancoragem das colunas D) — as colunas `delta_D*` seguem na escala original do 01_EDA.

    # 3. Engenharia de Grupo fitada EXCLUSIVAMENTE no Treino (Prevenção estrita de Leakage)
    novas_cols_grupo = []
    if 'card1' in train_df.columns and 'TransactionAmt' in train_df.columns:
        card1_stats = train_df.groupby("card1")["TransactionAmt"].mean()
        global_mean = train_df["TransactionAmt"].mean()

        for df_part in (train_df, test_df, oot_df):
            df_part["TransactionAmt_to_mean_card1"] = (
                df_part["TransactionAmt"] / (df_part["card1"].map(card1_stats).fillna(global_mean) + 1e-5)
            ).astype("float32")
        novas_cols_grupo.append("TransactionAmt_to_mean_card1")

    # 4. Frequency Encoding fitado EXCLUSIVAMENTE no Treino
    cols_to_encode = ["card1", "card2", "addr1", "P_emaildomain"]
    for col in cols_to_encode:
        if col in train_df.columns:
            freq_map = train_df[col].value_counts(dropna=False)
            for df_part in (train_df, test_df, oot_df):
                df_part[f"{col}_freq"] = df_part[col].map(freq_map).fillna(0).astype("float32")
            novas_cols_grupo.append(f"{col}_freq")

    # 5. Consolidação de Colunas (Remoção de IDs brutos que causam memorização)
    cols_memorizacao = {'TransactionID', 'isFraud', 'TransactionDT', 'card1', 'card2', 'addr1', 'addr2'}
    features_base = [c for c in train_df.columns if c not in cols_memorizacao]
    features_cols = list(dict.fromkeys(features_base + novas_cols_grupo))

    linear_cols = [c for c in meta.get("linear_baseline_features", []) if c in features_cols]
    cat_cols = train_df[features_cols].select_dtypes(include=['object', 'category']).columns.tolist()

    X_train = train_df[features_cols].copy()
    X_test  = test_df[features_cols].copy()
    X_oot   = oot_df[features_cols].copy()

    # Fallback defensivo para linear_cols se não houver lista salva em metadata
    if not linear_cols:
        linear_cols = [c for c in X_train.select_dtypes(include=[np.number]).columns if c not in cols_memorizacao][:50]

    # 6. Ordinal Encoding fitado APENAS no Treino (evita leakage de categorias futuras)
    if cat_cols:
        cat_train_df = pd.DataFrame(index=train_df.index)
        cat_test_df  = pd.DataFrame(index=test_df.index)
        cat_oot_df   = pd.DataFrame(index=oot_df.index)
        for c in cat_cols:
            cat_train_df[c] = train_df[c].astype(object).fillna("Missing").astype(str)
            cat_test_df[c]  = test_df[c].astype(object).fillna("Missing").astype(str)
            cat_oot_df[c]   = oot_df[c].astype(object).fillna("Missing").astype(str)
            train_df[c]     = cat_train_df[c]
            test_df[c]      = cat_test_df[c]
            oot_df[c]       = cat_oot_df[c]

        ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train[cat_cols] = ord_enc.fit_transform(cat_train_df[cat_cols]).astype("int32")
        X_test[cat_cols]  = ord_enc.transform(cat_test_df[cat_cols]).astype("int32")
        X_oot[cat_cols]   = ord_enc.transform(cat_oot_df[cat_cols]).astype("int32")

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test  = X_test.replace([np.inf, -np.inf], np.nan)
    X_oot   = X_oot.replace([np.inf, -np.inf], np.nan)

    y_train = train_df["isFraud"].astype('int8')
    y_test  = test_df["isFraud"].astype('int8')
    y_oot   = oot_df["isFraud"].astype('int8')

    # 7. Taxa de Desbalanceamento e Blocos Mensais (GroupKFold apenas no Treino)
    taxa_desbalanceamento = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    grupos_mes = (train_df["TransactionDT"] // (3600 * 24 * 30)).astype(int).values

    print("-" * 65)
    print(f"X_train shape ({train_ratio*100:.0f}%): {X_train.shape} | Taxa de Fraude: {y_train.mean()*100:.2f}%")
    print(f"X_test  shape ({test_ratio*100:.0f}%): {X_test.shape}  | Taxa de Fraude: {y_test.mean()*100:.2f}%")
    print(f"X_oot   shape ({(1-train_ratio-test_ratio)*100:.0f}%): {X_oot.shape}  | Taxa de Fraude: {y_oot.mean()*100:.2f}%  [BLINDADO P/ AVALIAÇÃO FINAL]")
    print(f"Total de features preditoras: {len(features_cols)}")
    print(f"Features lineares (LR): {len(linear_cols)}")
    print(f"Taxa de desbalanceamento (Treino): {taxa_desbalanceamento:.2f}:1")
    print(f"Blocos mensais no Treino (GroupKFold): {np.unique(grupos_mes).tolist()} (Total: {len(np.unique(grupos_mes))} meses)")
    print("-" * 65)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_oot': X_oot,
        'y_train': y_train,
        'y_test': y_test,
        'y_oot': y_oot,
        'features_cols': features_cols,
        'linear_cols': linear_cols,
        'cat_cols': cat_cols,
        'taxa_desbalanceamento': taxa_desbalanceamento,
        'grupos_mes': grupos_mes,
        'train_df': train_df,
        'test_df': test_df,
        'oot_df': oot_df,
    }


def obter_amt_real(df: pd.DataFrame, col: str = 'TransactionAmt') -> np.ndarray:
    """Retorna TransactionAmt em US$ reais (reverte log1p se a coluna estiver em escala log)."""
    valores = df[col].values.astype(float)
    return np.expm1(valores) if df[col].mean() < 10 else valores


def blocos_mensais(df: pd.DataFrame, col_dt: str = 'TransactionDT') -> np.ndarray:
    """Índice do bloco mensal (30 dias) de cada transação."""
    return (df[col_dt] // (3600 * 24 * 30)).astype(int).values


# ==============================================================================
# PIPELINE REPRODUTÍVEL (scripts/): CARGA DAS BASES BRUTAS IEEE-CIS
# ==============================================================================
ARQUIVOS_TREINO = ('train_transaction.csv', 'train_identity.csv')
ARQUIVOS_TESTE = ('test_transaction.csv', 'test_identity.csv')


def garantir_bases_brutas(
    raw_dir,
    arquivos=ARQUIVOS_TREINO,
    auto_download: bool = False,
    competicao: str = 'ieee-fraud-detection',
    forcar_download: bool = False,
):
    """
    Garante que os CSVs brutos existam em `raw_dir`. Baixa via KaggleHub quando
    `auto_download=True` (requer login/termos aceitos na competição); caso contrário
    falha com instrução clara.
    """
    from pathlib import Path

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    faltantes = [a for a in arquivos if not (raw_dir / a).exists()]
    if not faltantes and not forcar_download:
        return raw_dir
    if not auto_download and not forcar_download:
        raise FileNotFoundError(
            f"Arquivos brutos ausentes em {raw_dir}: {faltantes}. Copie-os da competição "
            f"'{competicao}' ou use `--set paths.auto_download_kaggle=true`."
        )
    import kagglehub
    print(f"[KaggleHub] Baixando a competição '{competicao}' para {raw_dir} ...")
    kagglehub.competition_download(competicao, force_download=forcar_download, output_dir=str(raw_dir))
    faltantes = [a for a in arquivos if not (raw_dir / a).exists()]
    if faltantes:
        raise FileNotFoundError(f"Download concluído, mas ainda faltam: {faltantes} em {raw_dir}.")
    return raw_dir


def carregar_bases_brutas(raw_dir, prefixo: str = 'train'):
    """Lê `<prefixo>_transaction.csv` e `<prefixo>_identity.csv` de `raw_dir`."""
    from pathlib import Path

    raw_dir = Path(raw_dir)
    df_tr = pd.read_csv(raw_dir / f'{prefixo}_transaction.csv')
    df_id = pd.read_csv(raw_dir / f'{prefixo}_identity.csv')
    print(f"[Raw] {prefixo}_transaction: {df_tr.shape} | {prefixo}_identity: {df_id.shape}")
    return df_tr, df_id
