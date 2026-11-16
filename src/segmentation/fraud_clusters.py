"""Pipeline de segmentação das fraudes: preparo, PCA/UMAP, K-Means/DBSCAN/HDBSCAN, arquétipos e cache.

Código migrado de `utils.py` (fachada de compatibilidade dos notebooks).
"""
from __future__ import annotations

from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple, Any, Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.data.preprocess import (  # noqa: F401
    reverter_escala_log1p,
)


def preparar_dados_clusterizacao(
    df_fraude: pd.DataFrame,
    meta: Optional[Dict[str, Any]] = None,
    cols_id_tempo_remover: Optional[List[str]] = None,
    cols_cat_baixa_card: Optional[List[str]] = None,
    cols_cat_alta_card: Optional[List[str]] = None,
    max_ohe_cardinalidade: int = 15,
    aplicar_log1p_assimetricas: bool = True,
    limiar_skew_log: float = 1.0,
    corrigir_drift_d: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame, List[str], Any]:
    """Prepara matriz numérica limpa e escalada (RobustScaler) para clusterização de fraudes.

    O estado ajustado (mapas de frequência, dummies, colunas em log1p e ordem das features)
    fica salvo em `preprocessador.estado_clusterizacao_`, permitindo aplicar a mesma
    transformação a novas fraudes via `aplicar_preparo_clusterizacao`.
    `corrigir_drift_d` é mantido apenas por compatibilidade e é ignorado (sem ancoragem das colunas D).

    Etapas aplicadas:
    1. (Sem ancoragem das colunas D — `delta_D*` seguem na escala original do 01_EDA.)
    2. Remoção de IDs brutos e timestamps arbitrários (`TransactionID`, `TransactionDT`,
       `card1`, `card2`, `addr1`, `addr2`, além de `isFraud`).
    3. Codificação de categóricas: `pd.get_dummies(..., dtype=float)` para baixa cardinalidade
       (ex: `ProductCD`, `card4`, `card6_limpo`, `M1_M2_M3_assinatura`) e Frequency Encoding
       para alta cardinalidade (ex: `P_emaildomain`, `device_brand`).
    4. Aplicação de `np.log1p` apenas nas numéricas estritamente não-negativas que ainda
       apresentem assimetria > `limiar_skew_log` (evita duplo log após o 01_EDA).
    5. Imputação por mediana (`SimpleImputer`) + escala robusta (`RobustScaler`).
    """
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler

    df_work = df_fraude.copy()

    # 2. Remoção de IDs brutos, timestamps e target
    padrao_remover = ["TransactionID", "isFraud", "TransactionDT", "card1", "card2", "addr1", "addr2"]
    lista_remover = cols_id_tempo_remover if cols_id_tempo_remover is not None else padrao_remover
    cols_para_dropar = [c for c in lista_remover if c in df_work.columns]
    if cols_para_dropar:
        df_work = df_work.drop(columns=cols_para_dropar)

    # 3. Separação e codificação de categóricas
    cols_cat_presentes = list(df_work.select_dtypes(include=["object", "category"]).columns)
    if meta and "cat_features" in meta:
        meta_cats = [c for c in meta["cat_features"] if c in df_work.columns and c not in cols_cat_presentes]
        cols_cat_presentes.extend(meta_cats)

    alta_card_padrao = {"P_emaildomain", "device_brand", "P_email_suffix", "browser_family"}
    if cols_cat_alta_card is not None:
        cols_alta = [c for c in cols_cat_alta_card if c in cols_cat_presentes]
    else:
        cols_alta = [
            c for c in cols_cat_presentes
            if c in alta_card_padrao or df_work[c].nunique(dropna=True) > max_ohe_cardinalidade
        ]

    if cols_cat_baixa_card is not None:
        cols_baixa = [c for c in cols_cat_baixa_card if c in cols_cat_presentes and c not in cols_alta]
    else:
        cols_baixa = [c for c in cols_cat_presentes if c not in cols_alta]

    # Frequency encoding para alta cardinalidade (cria/atualiza versão normalizada no recorte)
    cols_freq_criadas: List[str] = []
    mapas_freq: Dict[str, Dict[str, float]] = {}
    for col in cols_alta:
        serie_str = df_work[col].astype("object").fillna("missing").astype(str)
        freqs = serie_str.value_counts(normalize=True)
        mapas_freq[col] = freqs.to_dict()
        nome_freq = f"{col}_cluster_freq"
        df_work[nome_freq] = serie_str.map(freqs).astype("float64")
        cols_freq_criadas.append(nome_freq)
        df_work = df_work.drop(columns=[col])

    # One-Hot Encoding para baixa cardinalidade
    cols_ohe_criadas: List[str] = []
    if cols_baixa:
        for col in cols_baixa:
            df_work[col] = df_work[col].astype("object").fillna("missing").astype(str)
        antes_ohe = set(df_work.columns)
        df_work = pd.get_dummies(df_work, columns=cols_baixa, drop_first=False, dtype=float)
        cols_ohe_criadas = sorted(list(set(df_work.columns) - antes_ohe))

    # Garante apenas colunas numéricas e limpa infinitos
    df_work = df_work.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)

    # Remove colunas totalmente nulas ou de variância zero
    cols_validas = [c for c in df_work.columns if df_work[c].notna().any() and df_work[c].nunique(dropna=True) > 1]
    df_work = df_work[cols_validas].copy()

    # 4. Log1p em numéricas contínuas/assimétricas estritamente não-negativas
    cols_log1p_aplicadas: List[str] = []
    if aplicar_log1p_assimetricas:
        set_ohe = set(cols_ohe_criadas)
        for col in df_work.columns:
            if col in set_ohe:
                continue
            serie = df_work[col].dropna()
            if len(serie) < 10 or serie.nunique() <= 2:
                continue
            if serie.min() >= 0 and float(serie.skew()) > limiar_skew_log:
                df_work[col] = np.log1p(df_work[col].astype("float64"))
                cols_log1p_aplicadas.append(col)

    # 5. Imputação por mediana + RobustScaler
    feature_names = list(df_work.columns)
    preprocessador = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])
    X_tratado = preprocessador.fit_transform(df_work)

    # Estado ajustado no recorte de referência (reaplicável a novas fraudes sem re-ajuste)
    preprocessador.estado_clusterizacao_ = {
        "cols_remover": list(lista_remover),
        "cols_alta": list(cols_alta),
        "mapas_freq": mapas_freq,
        "cols_baixa": list(cols_baixa),
        "cols_ohe": list(cols_ohe_criadas),
        "cols_log1p": list(cols_log1p_aplicadas),
        "feature_names": list(feature_names),
    }

    print("Pré-processamento de clusterização concluído:")
    print(f" - Matriz final escalada: {X_tratado.shape[0]:,} linhas x {X_tratado.shape[1]} features")
    print(f" - Dummies (baixa cardinalidade): {len(cols_ohe_criadas)} colunas de {len(cols_baixa)} originais")
    print(f" - Frequency encoding (alta card.): {len(cols_freq_criadas)} colunas {cols_alta}")
    print(f" - Log1p adicional (skew > {limiar_skew_log}): {len(cols_log1p_aplicadas)} colunas")

    return X_tratado, df_work, feature_names, preprocessador


def aplicar_preparo_clusterizacao(df_novo: pd.DataFrame, preprocessador: Any) -> np.ndarray:
    """Aplica a novas fraudes (ex.: Validação/OOT) o mesmo pré-processamento ajustado em
    `preparar_dados_clusterizacao` (mapas de frequência, dummies, log1p, imputação e escala),
    sem re-ajustar nada. Categorias inéditas recebem frequência 0 e dummies zeradas."""
    estado = preprocessador.estado_clusterizacao_
    df_work = df_novo.drop(columns=[c for c in estado["cols_remover"] if c in df_novo.columns])

    for col in estado["cols_alta"]:
        if col in df_work.columns:
            serie_str = df_work[col].astype("object").fillna("missing").astype(str)
        else:
            serie_str = pd.Series("missing", index=df_work.index)
        df_work[f"{col}_cluster_freq"] = serie_str.map(estado["mapas_freq"][col]).fillna(0.0).astype("float64")
        df_work = df_work.drop(columns=[col], errors="ignore")

    cols_baixa = [c for c in estado["cols_baixa"] if c in df_work.columns]
    for col in cols_baixa:
        df_work[col] = df_work[col].astype("object").fillna("missing").astype(str)
    if cols_baixa:
        df_work = pd.get_dummies(df_work, columns=cols_baixa, drop_first=False, dtype=float)

    df_work = df_work.reindex(columns=estado["feature_names"])
    cols_ohe = [c for c in estado["cols_ohe"] if c in df_work.columns]
    df_work[cols_ohe] = df_work[cols_ohe].fillna(0.0)
    df_work = df_work.astype("float64").replace([np.inf, -np.inf], np.nan)

    for col in estado["cols_log1p"]:
        df_work[col] = np.log1p(df_work[col].clip(lower=0))

    return preprocessador.transform(df_work)


def recortar_treino_temporal(df_master: pd.DataFrame, train_ratio: float = 0.60) -> pd.DataFrame:
    """Retorna o bloco de Treino (primeiros `train_ratio` cronológicos por `TransactionDT`),
    com o mesmo corte de `preparar_dados_modelagem` (Validação e OOT ficam de fora)."""
    df_sorted = df_master.sort_values("TransactionDT").reset_index(drop=True)
    return df_sorted.iloc[:int(len(df_sorted) * train_ratio)].copy()


def projetar_pca(
    X_tratado: np.ndarray,
    n_components: int = 10,
    random_state: int = 42,
) -> Tuple[np.ndarray, Any]:
    """Projeta a matriz escalada em `n_components` componentes principais (PCA)."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_tratado)
    var_acum = float(pca.explained_variance_ratio_.sum() * 100)
    print(f"PCA ({n_components}D) concluído | Variância explicada acumulada: {var_acum:.2f}%")
    return X_pca, pca


def projetar_umap(
    X_tratado: np.ndarray,
    n_components: int = 10,
    n_neighbors: int = 40,
    min_dist: float = 0.0,
    metric: str = "euclidean",
    random_state: int = 42,
) -> Tuple[np.ndarray, Any]:
    """Projeta a matriz escalada via UMAP (10D para clusterizar com min_dist=0.0 ou 2D para plotar)."""
    try:
        import umap
    except ImportError as exc:
        raise ImportError("Pacote 'umap-learn' não encontrado. Instale com `pip install umap-learn`.") from exc

    modelo_umap = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    X_umap = modelo_umap.fit_transform(X_tratado)
    print(f"UMAP ({n_components}D | n_neighbors={n_neighbors}, min_dist={min_dist}) concluído: shape {X_umap.shape}")
    return X_umap, modelo_umap


def curva_elbow(
    X_tratado: np.ndarray,
    k_range: Iterable[int] = range(2, 13),
    n_init: int = 10,
    max_iter: int = 300,
    random_state: int = 42,
) -> Dict[int, float]:
    """Calcula a inércia (SSE) do K-Means ao longo de uma faixa de K (adaptado do Projeto-07)."""
    from sklearn.cluster import KMeans

    inercias: Dict[int, float] = {}
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=n_init, max_iter=max_iter, random_state=random_state)
        km.fit(X_tratado)
        inercias[k] = float(km.inertia_)
    return inercias


def avaliar_kmeans_range(
    X_tratado: np.ndarray,
    k_range: Iterable[int] = range(2, 13),
    sample_size: Optional[int] = 10000,
    n_init: int = 10,
    max_iter: int = 300,
    random_state: int = 42,
) -> pd.DataFrame:
    """Varre K-Means para `k_range` calculando Inércia, Silhouette, Davies-Bouldin e Calinski-Harabasz."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    resultados = []
    n_amostras = len(X_tratado)
    sil_sample = sample_size if (sample_size is not None and n_amostras > sample_size) else None

    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=n_init, max_iter=max_iter, random_state=random_state)
        labels = km.fit_predict(X_tratado)
        resultados.append({
            "K": int(k),
            "Inercia": float(km.inertia_),
            "Silhouette": float(silhouette_score(X_tratado, labels, sample_size=sil_sample, random_state=random_state)),
            "Davies_Bouldin": float(davies_bouldin_score(X_tratado, labels)),
            "Calinski_Harabasz": float(calinski_harabasz_score(X_tratado, labels)),
        })

    return pd.DataFrame(resultados)


def avaliar_grade_dbscan(
    X_tratado: np.ndarray,
    grade_eps: Iterable[float],
    grade_min_samples: Iterable[int],
    sample_size: Optional[int] = 10000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Executa busca em grade para DBSCAN calculando métricas internas estritamente sem o ruído (-1)."""
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    resultados = []
    n_amostras = len(X_tratado)

    for eps in grade_eps:
        for ms in grade_min_samples:
            modelo = DBSCAN(eps=eps, min_samples=ms)
            labels = modelo.fit_predict(X_tratado)

            mask = labels != -1
            n_ruido = int((~mask).sum())
            pct_ruido = float(n_ruido / n_amostras * 100)
            clusters_validos = np.unique(labels[mask])
            n_clusters = int(len(clusters_validos))

            if n_clusters >= 2 and int(mask.sum()) > n_clusters:
                X_val = X_tratado[mask]
                y_val = labels[mask]
                sil_sample = min(sample_size, len(X_val)) if (sample_size is not None and len(X_val) > sample_size) else None
                sil = float(silhouette_score(X_val, y_val, sample_size=sil_sample, random_state=random_state))
                db = float(davies_bouldin_score(X_val, y_val))
                ch = float(calinski_harabasz_score(X_val, y_val))
            else:
                sil, db, ch = np.nan, np.nan, np.nan

            resultados.append({
                "eps": float(eps),
                "min_samples": int(ms),
                "N_Clusters": n_clusters,
                "N_Ruido": n_ruido,
                "%_Ruido": round(pct_ruido, 2),
                "Silhouette": sil,
                "Davies_Bouldin": db,
                "Calinski_Harabasz": ch,
            })

    return pd.DataFrame(resultados)


def avaliar_grade_hdbscan(
    X_tratado: np.ndarray,
    grade_min_cluster_size: Iterable[int] = (100, 150, 250, 400),
    grade_min_samples: Iterable[int] = (15, 30),
    gen_min_span_tree: bool = True,
    prediction_data: bool = False,
    sample_size: Optional[int] = 10000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Executa grade do HDBSCAN extraindo DBCV (`relative_validity_`) e métricas sem ruído (-1)."""
    try:
        import hdbscan
    except ImportError as exc:
        raise ImportError("Pacote 'hdbscan' não encontrado. Instale com `pip install hdbscan`.") from exc

    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    resultados = []
    n_amostras = len(X_tratado)

    for mcs in grade_min_cluster_size:
        for ms in grade_min_samples:
            modelo = hdbscan.HDBSCAN(
                min_cluster_size=int(mcs),
                min_samples=int(ms),
                gen_min_span_tree=gen_min_span_tree,
                prediction_data=prediction_data,
            )
            labels = modelo.fit_predict(X_tratado)

            mask = labels != -1
            n_ruido = int((~mask).sum())
            pct_ruido = float(n_ruido / n_amostras * 100)
            clusters_validos = np.unique(labels[mask])
            n_clusters = int(len(clusters_validos))
            dbcv = float(getattr(modelo, "relative_validity_", np.nan)) if gen_min_span_tree else np.nan

            if n_clusters >= 2 and int(mask.sum()) > n_clusters:
                X_val = X_tratado[mask]
                y_val = labels[mask]
                sil_sample = min(sample_size, len(X_val)) if (sample_size is not None and len(X_val) > sample_size) else None
                sil = float(silhouette_score(X_val, y_val, sample_size=sil_sample, random_state=random_state))
                db = float(davies_bouldin_score(X_val, y_val))
                ch = float(calinski_harabasz_score(X_val, y_val))
            else:
                sil, db, ch = np.nan, np.nan, np.nan

            resultados.append({
                "min_cluster_size": int(mcs),
                "min_samples": int(ms),
                "N_Clusters": n_clusters,
                "N_Ruido": n_ruido,
                "%_Ruido": round(pct_ruido, 2),
                "DBCV": round(dbcv, 4) if not np.isnan(dbcv) else np.nan,
                "Silhouette": sil,
                "Davies_Bouldin": db,
                "Calinski_Harabasz": ch,
            })

    return pd.DataFrame(resultados)


def avaliar_rotulos_clusterizacao(
    labels: np.ndarray,
    X_espaco: np.ndarray,
    nome_modelo: str,
    nome_espaco: str,
    dbcv: float = np.nan,
    sample_size: Optional[int] = 10000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Avalia um vetor de rótulos filtrando estritamente ruído (-1) para a tabela comparativa."""
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    labels_arr = np.asarray(labels)
    mask = labels_arr != -1
    n_total = len(labels_arr)
    n_ruido = int((~mask).sum())
    pct_ruido = round(float(n_ruido / n_total * 100), 2)
    n_clusters = int(len(np.unique(labels_arr[mask])))

    if n_clusters >= 2 and int(mask.sum()) > n_clusters:
        X_val = X_espaco[mask]
        y_val = labels_arr[mask]
        sil_sample = min(sample_size, len(X_val)) if (sample_size is not None and len(X_val) > sample_size) else None
        sil = round(float(silhouette_score(X_val, y_val, sample_size=sil_sample, random_state=random_state)), 4)
        db = round(float(davies_bouldin_score(X_val, y_val)), 4)
        ch = round(float(calinski_harabasz_score(X_val, y_val)), 2)
    else:
        sil, db, ch = np.nan, np.nan, np.nan

    return {
        "Algoritmo": nome_modelo,
        "Espaco": nome_espaco,
        "N_Clusters": n_clusters,
        "%_Ruido": pct_ruido,
        "Silhouette": sil,
        "Davies_Bouldin": db,
        "Calinski_Harabasz": ch,
        "DBCV": round(float(dbcv), 4) if not np.isnan(dbcv) else np.nan,
    }


def _resolver_colunas_perfil_fraude(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Seleciona colunas comportamentais para perfilamento, suportando nomes atuais (01_EDA) e legados."""
    def _primeiro(opcoes: Iterable[str]) -> List[str]:
        return next(([c] for c in opcoes if c in df.columns), [])

    def _varios_unicos(opcoes: Iterable[str], limite: int) -> List[str]:
        escolhidas: List[str] = []
        vistos = set()
        for c in opcoes:
            if c in df.columns and c not in vistos:
                escolhidas.append(c)
                vistos.add(c)
                if len(escolhidas) >= limite:
                    break
        return escolhidas

    cols_moda = (
        _primeiro(["ProductCD"])
        + _primeiro(["card4"])
        + _primeiro(["cartao_tipo_cat", "card6_limpo"])
        + _primeiro(["disp_marca", "device_brand"])
        + _primeiro(["P_emaildomain"])
    )
    cols_nulos = _primeiro(["cont_null_tr", "tr_null_count"]) + _primeiro(["cont_null_id", "id_null_count"])
    cols_c = _varios_unicos(
        ["vel_C1", "vel_C2", "vel_C5", "vel_C8", "vel_C13", "vel_C14"]
        + [f"vel_C{i}" for i in range(1, 15)]
        + ["C1", "C2", "C5", "C8", "C13", "C14"]
        + [f"C{i}" for i in range(1, 15)],
        limite=5,
    )
    cols_d = _varios_unicos(
        ["delta_D1", "delta_D2", "delta_D4", "delta_D10", "delta_D15"]
        + [f"delta_D{i}" for i in range(1, 16)]
        + ["D1", "D2", "D4", "D10", "D15"]
        + [f"D{i}" for i in range(1, 16)],
        limite=4,
    )
    cols_extras = (
        _primeiro(["valor_razao_media_card1", "TransactionAmt_to_mean_card1"])
        + _primeiro(["match_cont_T", "M_true_count"])
        + _primeiro(["match_cont_F"])
        + _primeiro(["tempo_hora"])
    )
    return {
        "moda": cols_moda,
        "nulos": cols_nulos,
        "vel": cols_c,
        "delta": cols_d,
        "extras": cols_extras,
    }


def gerar_perfil_fraude_clusters(
    df_fraude: pd.DataFrame,
    col_cluster: str = "cluster_campeao",
    ignorar_ruido: bool = False,
) -> pd.DataFrame:
    """Gera tabela de perfilamento de negócio por cluster: financeiro, modas, nulos, velocidade (C), deltas (D) e validações."""
    df = df_fraude.copy()
    if ignorar_ruido:
        df = df[df[col_cluster] != -1].copy()

    total_fraudes = len(df)
    amt_em_log = False
    if "TransactionAmt" in df.columns and pd.api.types.is_numeric_dtype(df["TransactionAmt"]):
        amt_valid = df["TransactionAmt"].dropna()
        if not amt_valid.empty and amt_valid.min() >= 0 and amt_valid.max() <= 20.0:
            amt_em_log = True
            df["_amt_usd"] = np.expm1(df["TransactionAmt"].astype("float64"))
        else:
            df["_amt_usd"] = df["TransactionAmt"].astype("float64")

    mapa_cols = _resolver_colunas_perfil_fraude(df)
    cols_moda = mapa_cols["moda"]
    cols_contagem_real = [
        c for c in mapa_cols["nulos"] + mapa_cols["vel"] + mapa_cols["delta"]
        if pd.api.types.is_numeric_dtype(df[c])
    ]
    cols_extras_num = [c for c in mapa_cols["extras"] if pd.api.types.is_numeric_dtype(df[c])]
    col_id_null = next((c for c in ["cont_null_id", "id_null_count"] if c in df.columns), None)

    # Reversão global consistente de log1p nas contagens/deltas antes de agrupar por cluster
    if cols_contagem_real:
        df, _ = reverter_escala_log1p(df, cols_monetarias=(), cols_contagem=cols_contagem_real, verbose=False)

    linhas = []
    for cid, grupo in df.groupby(col_cluster, sort=True):
        n_grupo = len(grupo)
        registro: Dict[str, Any] = {
            "Cluster": "Ruído (-1)" if cid == -1 else f"Cluster {cid}",
            "N_Fraudes": n_grupo,
            "%_Fraudes": round(n_grupo / total_fraudes * 100, 2),
        }

        if "_amt_usd" in grupo.columns:
            registro["Prejuizo_Total_USD"] = round(float(grupo["_amt_usd"].sum()), 2)
            registro["Ticket_Medio_USD"] = round(float(grupo["_amt_usd"].mean()), 2)
            registro["Ticket_Mediano_USD"] = round(float(grupo["_amt_usd"].median()), 2)

        for col in cols_moda:
            serie_str = grupo[col].astype("object").fillna("missing").astype(str)
            vc = serie_str.value_counts(normalize=True)
            if not vc.empty:
                registro[f"{col}_Moda"] = f"{vc.index[0]} ({vc.iloc[0]*100:.1f}%)"

        if col_id_null is not None:
            registro["%_Sem_Identidade"] = round(float(grupo[col_id_null].isna().mean() * 100), 1)

        for col in cols_contagem_real + cols_extras_num:
            registro[f"Media_{col}"] = round(float(grupo[col].astype("float64").mean()), 2)

        linhas.append(registro)

    if amt_em_log:
        print("Nota: `TransactionAmt` estava em escala `log1p`; valores USD foram reconvertidos via `expm1`.")
    return pd.DataFrame(linhas).sort_values("N_Fraudes", ascending=False).reset_index(drop=True)


def _avaliar_metricas_sem_ruido(
    labels: np.ndarray,
    X_espaco: np.ndarray,
    sample_size: Optional[int] = 10000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Calcula N_Clusters, %_Ruido e métricas internas filtrando estritamente `labels != -1`.

    Quando `len(X_val) > sample_size` (ex: 10.000), usa subsample estratificado aleatório
    apenas no cálculo do Silhouette Score para manter custo O(n^2) viável.
    """
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    labels_arr = np.asarray(labels)
    mask = labels_arr != -1
    n_total = len(labels_arr)
    n_ruido = int((~mask).sum())
    pct_ruido = float(n_ruido / n_total * 100)
    n_clusters = int(len(np.unique(labels_arr[mask])))

    if n_clusters >= 2 and int(mask.sum()) > n_clusters:
        X_val = X_espaco[mask]
        y_val = labels_arr[mask]
        # Subsample de 10k (padrão) no Silhouette para evitar explosão O(n^2)
        sil_sample = min(sample_size, len(X_val)) if (sample_size is not None and len(X_val) > sample_size) else None
        sil = float(silhouette_score(X_val, y_val, sample_size=sil_sample, random_state=random_state))
        db = float(davies_bouldin_score(X_val, y_val))
        ch = float(calinski_harabasz_score(X_val, y_val))
    else:
        sil, db, ch = np.nan, np.nan, np.nan

    return {
        "n_clusters": n_clusters,
        "n_ruido": n_ruido,
        "pct_ruido": round(pct_ruido, 2),
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
    }


def _obter_umap_cache(
    cache_umap: Dict[Tuple[int, float], np.ndarray],
    X_tratado: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    n_components: int = 10,
    random_state: int = 42,
) -> np.ndarray:
    """Retorna projeção UMAP em cache para `(n_neighbors, min_dist)` evitando recálculos repetidos no Optuna."""
    import umap

    chave = (int(n_neighbors), round(float(min_dist), 3))
    if chave not in cache_umap:
        modelo = umap.UMAP(
            n_components=n_components,
            n_neighbors=chave[0],
            min_dist=chave[1],
            metric="euclidean",
            random_state=random_state,
        )
        cache_umap[chave] = modelo.fit_transform(X_tratado)
    return cache_umap[chave]


def otimizar_dbscan_optuna(
    X_espaco_fixo: Optional[np.ndarray] = None,
    X_tratado_umap: Optional[np.ndarray] = None,
    otimizar_umap: bool = False,
    n_trials: int = 25,
    max_ruido_pct: float = 45.0,
    sample_size: Optional[int] = 10000,
    random_state: int = 42,
    eps_range: Optional[Tuple[float, float]] = None,
    umap_neighbors_opcoes: Tuple[int, ...] = (25, 35, 45, 55),
    umap_min_dist_opcoes: Tuple[float, ...] = (0.0, 0.05, 0.1),
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, pd.DataFrame, Any]:
    """Otimiza DBSCAN via Optuna maximizando Silhouette (sem ruído -1, subsample=10k) com trava de ruído.

    Quando `otimizar_umap=True`, otimiza conjuntamente `n_neighbors` e `min_dist` do UMAP 10D
    (com cache de projeções e `eps` guiado por quantis da 15-NN k-distance do espaço corrente).
    """
    import optuna
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if otimizar_umap and X_tratado_umap is None:
        raise ValueError("Forneça `X_tratado_umap` quando `otimizar_umap=True`.")
    if not otimizar_umap and X_espaco_fixo is None:
        raise ValueError("Forneça `X_espaco_fixo` quando `otimizar_umap=False`.")

    cache_umap: Dict[Tuple[int, float], np.ndarray] = {}
    cache_kdists: Dict[Tuple[int, float], np.ndarray] = {}

    if not otimizar_umap and eps_range is None:
        viz = NearestNeighbors(n_neighbors=15).fit(X_espaco_fixo)
        dists, _ = viz.kneighbors(X_espaco_fixo)
        k_dists = np.sort(dists[:, 14])
        eps_range = (float(np.percentile(k_dists, 50)), float(np.percentile(k_dists, 98)))

    def objective(trial: optuna.Trial) -> float:
        if otimizar_umap:
            nn = trial.suggest_categorical("umap_n_neighbors", list(umap_neighbors_opcoes))
            md = trial.suggest_categorical("umap_min_dist", list(umap_min_dist_opcoes))
            chave = (int(nn), round(float(md), 3))
            X_cur = _obter_umap_cache(cache_umap, X_tratado_umap, nn, md, n_components=10, random_state=random_state)
            if chave not in cache_kdists:
                dists, _ = NearestNeighbors(n_neighbors=15).fit(X_cur).kneighbors(X_cur)
                cache_kdists[chave] = np.sort(dists[:, 14])
            eps_q = trial.suggest_float("eps_quantil", 0.55, 0.97)
            eps = float(np.quantile(cache_kdists[chave], eps_q))
        else:
            X_cur = X_espaco_fixo
            eps = trial.suggest_float("eps", eps_range[0], eps_range[1])

        ms = trial.suggest_int("min_samples", 10, 35, step=5)
        labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(X_cur)
        m = _avaliar_metricas_sem_ruido(labels, X_cur, sample_size=sample_size, random_state=random_state)

        trial.set_user_attr("eps_real", round(eps, 4))
        trial.set_user_attr("N_Clusters", m["n_clusters"])
        trial.set_user_attr("%_Ruido", m["pct_ruido"])
        trial.set_user_attr("Silhouette", m["silhouette"])
        trial.set_user_attr("Davies_Bouldin", m["davies_bouldin"])
        trial.set_user_attr("Calinski_Harabasz", m["calinski_harabasz"])

        if m["n_clusters"] < 2 or m["pct_ruido"] > max_ruido_pct or np.isnan(m["silhouette"]):
            return -1.0
        return m["silhouette"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    registros = []
    for t in study.trials:
        reg = {"trial": t.number, "objetivo": t.value, **t.params, **t.user_attrs}
        registros.append(reg)
    df_trials = pd.DataFrame(registros)

    best = study.best_trial
    best_params = dict(best.params)
    best_params["eps_real"] = best.user_attrs.get("eps_real", best_params.get("eps"))

    if otimizar_umap:
        X_best = _obter_umap_cache(
            cache_umap,
            X_tratado_umap,
            best_params["umap_n_neighbors"],
            best_params["umap_min_dist"],
            n_components=10,
            random_state=random_state,
        )
    else:
        X_best = X_espaco_fixo

    best_labels = DBSCAN(eps=float(best_params["eps_real"]), min_samples=int(best_params["min_samples"])).fit_predict(X_best)
    return best_params, best_labels, X_best, df_trials, study


def otimizar_hdbscan_optuna(
    X_espaco_fixo: Optional[np.ndarray] = None,
    X_tratado_umap: Optional[np.ndarray] = None,
    otimizar_umap: bool = False,
    n_trials: int = 25,
    max_ruido_pct: float = 45.0,
    sample_size: Optional[int] = 10000,
    random_state: int = 42,
    umap_neighbors_opcoes: Tuple[int, ...] = (25, 35, 45, 55),
    umap_min_dist_opcoes: Tuple[float, ...] = (0.0, 0.05, 0.1),
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, float, pd.DataFrame, Any]:
    """Otimiza HDBSCAN via Optuna maximizando DBCV (`relative_validity_`) com trava de viabilidade.

    Calcula Silhouette em subsample de `sample_size` (padrão 10k) estritamente sobre `labels != -1`.
    Quando `otimizar_umap=True`, otimiza conjuntamente `n_neighbors` e `min_dist` do UMAP 10D.
    """
    import hdbscan
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if otimizar_umap and X_tratado_umap is None:
        raise ValueError("Forneça `X_tratado_umap` quando `otimizar_umap=True`.")
    if not otimizar_umap and X_espaco_fixo is None:
        raise ValueError("Forneça `X_espaco_fixo` quando `otimizar_umap=False`.")

    cache_umap: Dict[Tuple[int, float], np.ndarray] = {}

    def objective(trial: optuna.Trial) -> float:
        if otimizar_umap:
            nn = trial.suggest_categorical("umap_n_neighbors", list(umap_neighbors_opcoes))
            md = trial.suggest_categorical("umap_min_dist", list(umap_min_dist_opcoes))
            X_cur = _obter_umap_cache(cache_umap, X_tratado_umap, nn, md, n_components=10, random_state=random_state)
        else:
            X_cur = X_espaco_fixo

        mcs = trial.suggest_int("min_cluster_size", 100, 400, step=25)
        ms = trial.suggest_int("min_samples", 10, 40, step=5)
        sel_method = trial.suggest_categorical("cluster_selection_method", ["eom", "leaf"])

        modelo = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            cluster_selection_method=sel_method,
            gen_min_span_tree=True,
        )
        labels = modelo.fit_predict(X_cur)
        dbcv = float(getattr(modelo, "relative_validity_", np.nan))
        m = _avaliar_metricas_sem_ruido(labels, X_cur, sample_size=sample_size, random_state=random_state)

        trial.set_user_attr("N_Clusters", m["n_clusters"])
        trial.set_user_attr("%_Ruido", m["pct_ruido"])
        trial.set_user_attr("DBCV", round(dbcv, 4) if not np.isnan(dbcv) else np.nan)
        trial.set_user_attr("Silhouette", m["silhouette"])
        trial.set_user_attr("Davies_Bouldin", m["davies_bouldin"])
        trial.set_user_attr("Calinski_Harabasz", m["calinski_harabasz"])

        if m["n_clusters"] < 2 or m["pct_ruido"] > max_ruido_pct or np.isnan(dbcv):
            return -1.0
        return dbcv

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    registros = []
    for t in study.trials:
        reg = {"trial": t.number, "objetivo": t.value, **t.params, **t.user_attrs}
        registros.append(reg)
    df_trials = pd.DataFrame(registros)

    best_params = dict(study.best_trial.params)
    if otimizar_umap:
        X_best = _obter_umap_cache(
            cache_umap,
            X_tratado_umap,
            best_params["umap_n_neighbors"],
            best_params["umap_min_dist"],
            n_components=10,
            random_state=random_state,
        )
    else:
        X_best = X_espaco_fixo

    modelo_best = hdbscan.HDBSCAN(
        min_cluster_size=int(best_params["min_cluster_size"]),
        min_samples=int(best_params["min_samples"]),
        cluster_selection_method=best_params["cluster_selection_method"],
        gen_min_span_tree=True,
    )
    best_labels = modelo_best.fit_predict(X_best)
    best_dbcv = float(getattr(modelo_best, "relative_validity_", np.nan))
    return best_params, best_labels, X_best, best_dbcv, df_trials, study


def consolidar_comparacao_clusterizacao(
    df_km_pca: pd.DataFrame,
    df_km_umap: pd.DataFrame,
    X_pca_10d: np.ndarray,
    X_umap_10d: np.ndarray,
    db_pca_res: Tuple[Dict[str, Any], np.ndarray, np.ndarray],
    db_umap_res: Tuple[Dict[str, Any], np.ndarray, np.ndarray],
    hdb_pca_res: Tuple[Dict[str, Any], np.ndarray, np.ndarray, float],
    hdb_umap_res: Tuple[Dict[str, Any], np.ndarray, np.ndarray, float],
    sample_size: Optional[int] = 10000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Consolida K-Means (PCA/UMAP) e campeões Optuna de DBSCAN/HDBSCAN em tabela única.

    Usa subsample de `sample_size` (padrão 10.000) no Silhouette Score e filtra ruído (-1).
    """
    from sklearn.cluster import KMeans

    candidatos_labels: Dict[str, np.ndarray] = {}
    registros: List[Dict[str, Any]] = []

    best_km_pca = df_km_pca.sort_values("Silhouette", ascending=False).iloc[0]
    lbl_km_pca = KMeans(n_clusters=int(best_km_pca["K"]), init="k-means++", n_init=10, random_state=random_state).fit_predict(X_pca_10d)
    candidatos_labels["KMeans_PCA"] = lbl_km_pca
    registros.append(avaliar_rotulos_clusterizacao(lbl_km_pca, X_pca_10d, f"K-Means (K={int(best_km_pca['K'])})", "PCA 10D", sample_size=sample_size, random_state=random_state))

    best_km_umap = df_km_umap.sort_values("Silhouette", ascending=False).iloc[0]
    lbl_km_umap = KMeans(n_clusters=int(best_km_umap["K"]), init="k-means++", n_init=10, random_state=random_state).fit_predict(X_umap_10d)
    candidatos_labels["KMeans_UMAP"] = lbl_km_umap
    registros.append(avaliar_rotulos_clusterizacao(lbl_km_umap, X_umap_10d, f"K-Means (K={int(best_km_umap['K'])})", "UMAP 10D", sample_size=sample_size, random_state=random_state))

    db_pca_p, db_pca_lbl, db_pca_X = db_pca_res
    candidatos_labels["DBSCAN_PCA"] = db_pca_lbl
    registros.append(avaliar_rotulos_clusterizacao(db_pca_lbl, db_pca_X, f"DBSCAN Optuna (eps={db_pca_p['eps_real']}, ms={db_pca_p['min_samples']})", "PCA 10D", sample_size=sample_size, random_state=random_state))

    db_umap_p, db_umap_lbl, db_umap_X = db_umap_res
    candidatos_labels["DBSCAN_UMAP"] = db_umap_lbl
    registros.append(avaliar_rotulos_clusterizacao(db_umap_lbl, db_umap_X, f"DBSCAN+UMAP Optuna (nn={db_umap_p['umap_n_neighbors']}, md={db_umap_p['umap_min_dist']}, eps={db_umap_p['eps_real']})", "UMAP 10D (Optuna)", sample_size=sample_size, random_state=random_state))

    hdb_pca_p, hdb_pca_lbl, hdb_pca_X, hdb_pca_dbcv = hdb_pca_res
    candidatos_labels["HDBSCAN_PCA"] = hdb_pca_lbl
    registros.append(avaliar_rotulos_clusterizacao(hdb_pca_lbl, hdb_pca_X, f"HDBSCAN Optuna (mcs={hdb_pca_p['min_cluster_size']}, ms={hdb_pca_p['min_samples']}, {hdb_pca_p['cluster_selection_method']})", "PCA 10D", dbcv=hdb_pca_dbcv, sample_size=sample_size, random_state=random_state))

    hdb_umap_p, hdb_umap_lbl, hdb_umap_X, hdb_umap_dbcv = hdb_umap_res
    candidatos_labels["HDBSCAN_UMAP"] = hdb_umap_lbl
    registros.append(avaliar_rotulos_clusterizacao(hdb_umap_lbl, hdb_umap_X, f"HDBSCAN+UMAP Optuna (nn={hdb_umap_p['umap_n_neighbors']}, md={hdb_umap_p['umap_min_dist']}, mcs={hdb_umap_p['min_cluster_size']})", "UMAP 10D (Optuna)", dbcv=hdb_umap_dbcv, sample_size=sample_size, random_state=random_state))

    return pd.DataFrame(registros), candidatos_labels


def gerar_tabela_arquetipos(
    df_fraude: pd.DataFrame,
    col_grupo: str = "arquetipo",
    ordem: Optional[List[str]] = None,
    col_cluster_origem: Optional[str] = "cluster",
) -> pd.DataFrame:
    """Tabela consolidada de totais por arquétipo (macrogrupo de clusters).

    Reaproveita `gerar_perfil_fraude_clusters` para financeiro (US$ reconvertido) e modas,
    adiciona os shares de volume/prejuízo e recalcula as médias de contagem (`C*`,
    `*_null_count`) na escala real com detecção global de `log1p`.
    """
    if col_grupo not in df_fraude.columns:
        raise KeyError(f"Coluna '{col_grupo}' não encontrada em df_fraude.")

    perfil = gerar_perfil_fraude_clusters(df_fraude, col_cluster=col_grupo, ignorar_ruido=False)
    perfil["Arquetipo"] = perfil["Cluster"].astype(str).str.replace(r"^Cluster ", "", regex=True)
    perfil = perfil.drop(columns="Cluster").set_index("Arquetipo")

    # Médias de contagem em escala real (detecção global, não por grupo)
    cols_media = [c for c in ["C1", "C2", "C13", "C14", "tr_null_count", "id_null_count"] if c in df_fraude.columns]
    df_real, _ = reverter_escala_log1p(df_fraude[[col_grupo] + cols_media], cols_monetarias=(), cols_contagem=cols_media, verbose=False)
    medias = df_real.groupby(col_grupo, observed=True)[cols_media].mean().round(2)
    for col in cols_media:
        perfil[f"Media_{col}"] = medias[col]

    if "id_null_count" in df_fraude.columns:
        perfil["%_Sem_Identidade"] = (
            df_fraude["id_null_count"].isna().groupby(df_fraude[col_grupo], observed=True).mean().mul(100).round(1)
        )

    if col_cluster_origem and col_cluster_origem in df_fraude.columns:
        perfil["Clusters_KMeans"] = (
            df_fraude.groupby(col_grupo, observed=True)[col_cluster_origem]
            .apply(lambda s: ", ".join(str(int(c)) for c in sorted(s.unique())))
        )

    if "Prejuizo_Total_USD" in perfil.columns:
        prejuizo_total = perfil["Prejuizo_Total_USD"].sum()
        perfil["%_Prejuizo_Total"] = (perfil["Prejuizo_Total_USD"] / prejuizo_total * 100).round(2)
        # Alavancagem > 1: o grupo gera mais prejuízo do que seu peso em quantidade sugere
        perfil["Alavancagem_Prejuizo_vs_Volume"] = (perfil["%_Prejuizo_Total"] / perfil["%_Fraudes"]).round(2)

    ordem_final = [g for g in (ordem or sorted(perfil.index)) if g in perfil.index]
    perfil = perfil.loc[ordem_final]

    colunas = [
        "Clusters_KMeans", "N_Fraudes", "%_Fraudes", "Prejuizo_Total_USD", "%_Prejuizo_Total",
        "Alavancagem_Prejuizo_vs_Volume", "Ticket_Medio_USD", "Ticket_Mediano_USD",
        "ProductCD_Moda", "cartao_tipo_cat_Moda", "card6_limpo_Moda",
        "disp_marca_Moda", "device_brand_Moda", "P_emaildomain_Moda",
        "%_Sem_Identidade", "Media_cont_null_tr", "Media_tr_null_count",
        "Media_cont_null_id", "Media_id_null_count",
        "Media_vel_C2", "Media_C2", "Media_vel_C8", "Media_vel_C12",
        "Media_delta_D1", "Media_D1", "Media_match_cont_T", "Media_tempo_hora",
    ]
    return perfil[[c for c in colunas if c in perfil.columns]].reset_index()


def _serie_diaria_arquetipos(
    df_fraude: pd.DataFrame,
    col_grupo: str,
    col_dt: str,
    ordem: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna (volume diário, prejuízo diário em US$) por arquétipo, com dias sem fraude = 0."""
    df_real, _ = reverter_escala_log1p(df_fraude[[col_grupo, col_dt, "TransactionAmt"]], cols_contagem=[], verbose=False)
    dt = df_real[col_dt].astype("float64")
    df_real["_dia"] = np.floor((dt - dt.min()) / 86400.0).astype(int)
    dias = pd.RangeIndex(0, int(df_real["_dia"].max()) + 1, name="Dia")

    volume = (
        df_real.groupby(["_dia", col_grupo], observed=True).size().unstack(fill_value=0)
        .reindex(index=dias, columns=ordem, fill_value=0)
    )
    prejuizo = (
        df_real.groupby(["_dia", col_grupo], observed=True)["TransactionAmt"].sum().unstack(fill_value=0.0)
        .reindex(index=dias, columns=ordem, fill_value=0.0)
    )
    return volume, prejuizo


def resumir_dinamica_temporal_arquetipos(
    df_fraude: pd.DataFrame,
    col_grupo: str = "arquetipo",
    col_dt: str = "TransactionDT",
    ordem: Optional[List[str]] = None,
    limiar_cv_rajada: float = 0.8,
    limiar_pico_mediana: float = 4.0,
    limiar_tendencia_mes_pct: float = 15.0,
) -> pd.DataFrame:
    """Métricas de dinâmica temporal diária por arquétipo e classificação do perfil.

    - `CV_Diario` (desvio/média) e `Pico/Mediana` medem rajadas (bursts);
    - `Tendencia_%_mes`: inclinação da regressão linear do volume diário, em % da média a cada 30 dias.
    Perfil: "Rajadas" se CV >= limiar_cv_rajada ou Pico/Mediana >= limiar_pico_mediana;
    senão "Crescimento"/"Retração" se |tendência| >= limiar_tendencia_mes_pct; senão "Estável".
    """
    ordem = ordem or sorted(df_fraude[col_grupo].dropna().unique())
    volume, prejuizo = _serie_diaria_arquetipos(df_fraude, col_grupo, col_dt, ordem)
    dias = volume.index.values.astype("float64")

    registros = []
    for g in ordem:
        s = volume[g].astype("float64")
        media = float(s.mean())
        mediana = float(s.median())
        cv = float(s.std(ddof=0) / media) if media > 0 else np.nan
        pico_mediana = float(s.max() / mediana) if mediana > 0 else np.inf
        inclinacao = float(np.polyfit(dias, s.values, 1)[0]) if media > 0 else 0.0
        tendencia = inclinacao * 30 / media * 100 if media > 0 else np.nan

        if cv >= limiar_cv_rajada or pico_mediana >= limiar_pico_mediana:
            perfil = "Rajadas (burst)"
        elif tendencia >= limiar_tendencia_mes_pct:
            perfil = "Crescimento"
        elif tendencia <= -limiar_tendencia_mes_pct:
            perfil = "Retração"
        else:
            perfil = "Estável"

        registros.append({
            "Arquetipo": g,
            "Dias_Ativos_%": round(float((s > 0).mean() * 100), 1),
            "Media_Diaria": round(media, 1),
            "Mediana_Diaria": round(mediana, 1),
            "Pico_Diario": int(s.max()),
            "Dia_do_Pico": int(s.idxmax()),
            "Pico/Mediana": round(pico_mediana, 2),
            "CV_Diario": round(cv, 2),
            "Tendencia_%_mes": round(tendencia, 1),
            "Prejuizo_Medio_Diario_USD": round(float(prejuizo[g].mean()), 2),
            "Perfil_Temporal": perfil,
        })
    return pd.DataFrame(registros)


def verificar_disponibilidade_hdbscan(random_state: int = 42) -> Tuple[Optional[str], str]:
    """Verifica, com um **ajuste real** em mini-array, qual backend HDBSCAN funciona no ambiente.

    Apenas importar não basta: o pacote `hdbscan` pode importar e quebrar no `fit` por
    incompatibilidade de versão com o scikit-learn (ex.: `force_all_finite` removido no sklearn >= 1.8).
    Retorna `(backend, mensagem)`, com backend em {"hdbscan", "sklearn", None}:
    - "hdbscan": pacote original (expõe DBCV via `relative_validity_`);
    - "sklearn": `sklearn.cluster.HDBSCAN` (mesmo algoritmo, sem DBCV nativo);
    - None: nenhum disponível.
    """
    X_teste = np.random.default_rng(random_state).normal(size=(60, 3))
    falhas = []
    try:
        import hdbscan
        hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True).fit_predict(X_teste)
        versao = getattr(hdbscan, "__version__", "versão não informada")
        return "hdbscan", f"Backend 'hdbscan' operacional ({versao}) — DBCV (relative_validity_) disponível."
    except Exception as exc:  # ImportError, ABI do numpy ou API incompatível do sklearn
        falhas.append(f"hdbscan -> {type(exc).__name__}: {exc}")
    try:
        from sklearn.cluster import HDBSCAN as SkHDBSCAN
        SkHDBSCAN(min_cluster_size=5).fit_predict(X_teste)
        import sklearn
        return "sklearn", (
            f"Pacote 'hdbscan' inoperante ({falhas[0]}). Usando fallback sklearn.cluster.HDBSCAN "
            f"(sklearn {sklearn.__version__}) — sem DBCV nativo; critério de seleção = Silhouette."
        )
    except Exception as exc:
        falhas.append(f"sklearn.cluster.HDBSCAN -> {type(exc).__name__}: {exc}")
    return None, "Nenhum backend HDBSCAN disponível. Instale com `pip install hdbscan` ou atualize o scikit-learn (>= 1.3). " + " | ".join(falhas)


def ajustar_hdbscan(
    X_espaco: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_method: str = "eom",
    backend: str = "hdbscan",
) -> Tuple[np.ndarray, float, float]:
    """Ajusta o HDBSCAN no backend informado e retorna (rótulos, DBCV, tempo em segundos).

    DBCV só é exposto pelo pacote `hdbscan` (`relative_validity_`); no backend sklearn retorna NaN.
    """
    import time

    t0 = time.perf_counter()
    if backend == "hdbscan":
        import hdbscan
        modelo = hdbscan.HDBSCAN(
            min_cluster_size=int(min_cluster_size),
            min_samples=int(min_samples),
            cluster_selection_method=cluster_selection_method,
            gen_min_span_tree=True,
        )
        labels = modelo.fit_predict(X_espaco)
        dbcv = float(getattr(modelo, "relative_validity_", np.nan))
    elif backend == "sklearn":
        from sklearn.cluster import HDBSCAN as SkHDBSCAN
        labels = SkHDBSCAN(
            min_cluster_size=int(min_cluster_size),
            min_samples=int(min_samples),
            cluster_selection_method=cluster_selection_method,
        ).fit_predict(X_espaco)
        dbcv = np.nan
    else:
        raise ValueError(f"Backend HDBSCAN inválido ou indisponível: {backend!r}")
    return np.asarray(labels), dbcv, round(time.perf_counter() - t0, 2)


def avaliar_viabilidade_hdbscan(
    X_espaco: np.ndarray,
    grade_min_cluster_size: Iterable[int] = (100, 200, 400),
    grade_min_samples: Iterable[int] = (10, 30),
    max_ruido_pct: float = 35.0,
    min_clusters: int = 3,
    max_clusters: int = 20,
    backend: str = "hdbscan",
    sample_size: Optional[int] = 10000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Grade enxuta do HDBSCAN com tempo de execução por configuração e critérios de viabilidade.

    No backend "hdbscan" reaproveita `avaliar_grade_hdbscan` (uma chamada por combinação, para
    cronometrar cada ajuste); no backend "sklearn" usa `ajustar_hdbscan` + `_avaliar_metricas_sem_ruido`.
    Uma configuração é **viável** quando `%_Ruido <= max_ruido_pct` e
    `min_clusters <= N_Clusters <= max_clusters` (evita colapso em 1-2 blobs ou pulverização em
    micro-clusters sem leitura de negócio).
    """
    import time

    linhas = []
    for mcs in grade_min_cluster_size:
        for ms in grade_min_samples:
            t0 = time.perf_counter()
            if backend == "hdbscan":
                linha = avaliar_grade_hdbscan(
                    X_espaco, grade_min_cluster_size=(int(mcs),), grade_min_samples=(int(ms),),
                    gen_min_span_tree=True, sample_size=sample_size, random_state=random_state,
                ).iloc[0].to_dict()
            else:
                labels, dbcv, _ = ajustar_hdbscan(X_espaco, int(mcs), int(ms), backend=backend)
                m = _avaliar_metricas_sem_ruido(labels, X_espaco, sample_size=sample_size, random_state=random_state)
                linha = {
                    "min_cluster_size": int(mcs), "min_samples": int(ms),
                    "N_Clusters": m["n_clusters"], "N_Ruido": m["n_ruido"], "%_Ruido": m["pct_ruido"],
                    "DBCV": dbcv, "Silhouette": m["silhouette"],
                    "Davies_Bouldin": m["davies_bouldin"], "Calinski_Harabasz": m["calinski_harabasz"],
                }
            linha["Tempo_s"] = round(time.perf_counter() - t0, 2)

            motivos = []
            if linha["%_Ruido"] > max_ruido_pct:
                motivos.append(f"ruído {linha['%_Ruido']:.1f}% > {max_ruido_pct:.0f}%")
            if linha["N_Clusters"] < min_clusters:
                motivos.append(f"colapso ({linha['N_Clusters']} clusters < {min_clusters})")
            if linha["N_Clusters"] > max_clusters:
                motivos.append(f"micro-clusters ({linha['N_Clusters']} > {max_clusters})")
            linha["Viavel"] = not motivos
            linha["Motivo"] = "; ".join(motivos) if motivos else "OK"
            linhas.append(linha)

    df = pd.DataFrame(linhas)
    for c in ["Silhouette", "Davies_Bouldin"]:
        df[c] = df[c].astype(float).round(4)
    df["Calinski_Harabasz"] = df["Calinski_Harabasz"].astype(float).round(1)
    return df


def selecionar_config_hdbscan(
    df_viabilidade: pd.DataFrame,
    criterio: str = "DBCV",
) -> Tuple[pd.Series, bool]:
    """Seleciona a melhor configuração: maior `criterio` entre as viáveis (desempate: menor ruído).

    Se `criterio` estiver todo vazio (ex.: DBCV no backend sklearn), usa Silhouette.
    Se nenhuma configuração for viável, retorna a de menor `%_Ruido` com >= 2 clusters e `viavel=False`.
    """
    if criterio not in df_viabilidade.columns or df_viabilidade[criterio].isna().all():
        criterio = "Silhouette"
    viaveis = df_viabilidade[df_viabilidade["Viavel"]]
    if not viaveis.empty:
        melhor = viaveis.sort_values([criterio, "%_Ruido"], ascending=[False, True], na_position="last").iloc[0]
        return melhor, True
    candidatos = df_viabilidade[df_viabilidade["N_Clusters"] >= 2]
    base = candidatos if not candidatos.empty else df_viabilidade
    return base.sort_values(["%_Ruido", criterio], ascending=[True, False], na_position="last").iloc[0], False


def comparar_hdbscan_vs_arquetipos(
    labels_hdbscan: np.ndarray,
    arquetipos: Union[np.ndarray, pd.Series],
    labels_kmeans: Optional[np.ndarray] = None,
    ordem: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Concordância HDBSCAN x K-Means (arquétipos) e tabelas cruzadas.

    Retorna dict com:
    - `concordancia`: ARI/NMI do HDBSCAN (sem ruído -1) vs. 4 arquétipos e vs. K-Means K=10;
    - `crosstab`: contagem HDBSCAN (linhas, inclui ruído) x arquétipo (colunas), com totais;
    - `crosstab_pct`: % de cada cluster HDBSCAN em cada arquétipo + arquétipo dominante/pureza;
    - `ruido_por_arquetipo`: quanto de cada arquétipo o HDBSCAN descarta como ruído.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    hdb = np.asarray(labels_hdbscan)
    arq = pd.Series(arquetipos).astype(str).reset_index(drop=True)
    ordem = ordem or sorted(arq.unique())
    mask = hdb != -1

    linhas = []
    referencias = {"4 Arquétipos (K-Means agrupado)": arq.values}
    if labels_kmeans is not None:
        referencias["K-Means K=10 (clusters originais)"] = np.asarray(labels_kmeans)
    for nome, ref in referencias.items():
        if mask.sum() > 0 and len(np.unique(hdb[mask])) > 1:
            ari = adjusted_rand_score(ref[mask], hdb[mask])
            nmi = normalized_mutual_info_score(ref[mask], hdb[mask])
        else:
            ari, nmi = np.nan, np.nan
        linhas.append({
            "Referencia": nome,
            "Pontos_Comparados": int(mask.sum()),
            "Cobertura_%": round(float(mask.mean() * 100), 2),
            "ARI": round(float(ari), 4),
            "NMI": round(float(nmi), 4),
        })
    concordancia = pd.DataFrame(linhas)

    rotulo_hdb = pd.Series(np.where(hdb == -1, "Ruído (-1)", pd.Series(hdb).map(lambda c: f"HDB {c:02d}")), name="HDBSCAN")
    ct = pd.crosstab(rotulo_hdb, arq.rename("Arquetipo")).reindex(columns=ordem, fill_value=0)
    ordem_linhas = (["Ruído (-1)"] if "Ruído (-1)" in ct.index else []) + sorted(i for i in ct.index if i != "Ruído (-1)")
    ct = ct.loc[ordem_linhas]

    ct_pct = ct.div(ct.sum(axis=1), axis=0).mul(100).round(1)
    ct_pct["Arquetipo_Dominante"] = ct.idxmax(axis=1)
    ct_pct["Pureza_%"] = ct_pct[ordem].max(axis=1)
    ct_pct["N"] = ct.sum(axis=1)

    ct_total = ct.copy()
    ct_total["Total"] = ct_total.sum(axis=1)
    ct_total.loc["Total"] = ct_total.sum(axis=0)

    n_ruido_total = int((~mask).sum())
    ruido = pd.DataFrame({
        "N_Arquetipo": arq.value_counts().reindex(ordem).fillna(0).astype(int),
        "N_Ruido": arq[~mask].value_counts().reindex(ordem).fillna(0).astype(int),
    })
    ruido["%_do_Arquetipo_em_Ruido"] = (ruido["N_Ruido"] / ruido["N_Arquetipo"].replace(0, np.nan) * 100).round(1)
    ruido["%_do_Ruido_Total"] = (ruido["N_Ruido"] / max(n_ruido_total, 1) * 100).round(1)
    ruido.index.name = "Arquetipo"

    return {
        "concordancia": concordancia,
        "crosstab": ct_total,
        "crosstab_pct": ct_pct,
        "ruido_por_arquetipo": ruido.reset_index(),
    }


def veredito_viabilidade_hdbscan(
    metricas_hdbscan: Dict[str, Any],
    metricas_referencia: Dict[str, Any],
    tempo_s: float,
    max_ruido_pct: float = 35.0,
    min_clusters: int = 3,
    max_clusters: int = 20,
    max_tempo_s: float = 300.0,
) -> pd.DataFrame:
    """Checklist objetivo de viabilidade do HDBSCAN frente à referência (K-Means K=10).

    Critérios eliminatórios: ruído, nº de clusters e tempo. Critério de ganho (agregação):
    Silhouette (sem ruído) e Davies-Bouldin melhores que a referência no mesmo espaço.
    """
    sil_h, sil_r = metricas_hdbscan.get("Silhouette", np.nan), metricas_referencia.get("Silhouette", np.nan)
    db_h, db_r = metricas_hdbscan.get("Davies_Bouldin", np.nan), metricas_referencia.get("Davies_Bouldin", np.nan)
    n_cl = metricas_hdbscan.get("N_Clusters", 0)
    ruido = metricas_hdbscan.get("%_Ruido", 100.0)

    def _status(ok: bool) -> str:
        return "✅ Atende" if ok else "❌ Não atende"

    linhas = [
        ("Eliminatório", "% de ruído (-1)", f"{ruido:.1f}%", f"<= {max_ruido_pct:.0f}%", _status(ruido <= max_ruido_pct)),
        ("Eliminatório", "Nº de clusters", f"{n_cl}", f"{min_clusters}–{max_clusters}", _status(min_clusters <= n_cl <= max_clusters)),
        ("Eliminatório", "Tempo do ajuste final", f"{tempo_s:.1f} s", f"<= {max_tempo_s:.0f} s", _status(tempo_s <= max_tempo_s)),
        ("Ganho", "Silhouette (sem ruído) ↑", f"{sil_h:.4f}", f"> K-Means ({sil_r:.4f})", _status(bool(sil_h > sil_r))),
        ("Ganho", "Davies-Bouldin (sem ruído) ↓", f"{db_h:.4f}", f"< K-Means ({db_r:.4f})", _status(bool(db_h < db_r))),
    ]
    df = pd.DataFrame(linhas, columns=["Tipo", "Critério", "HDBSCAN", "Limite / Referência", "Status"])

    eliminatorios_ok = df.loc[df["Tipo"] == "Eliminatório", "Status"].str.startswith("✅").all()
    ganhos = int(df.loc[df["Tipo"] == "Ganho", "Status"].str.startswith("✅").sum())
    if not eliminatorios_ok:
        veredito = "INVIÁVEL — falha em critério eliminatório; manter K-Means (4 arquétipos) como segmentação oficial."
    elif ganhos == 2:
        veredito = "VIÁVEL E AGREGADOR — supera o K-Means nas métricas internas; avaliar uso complementar."
    else:
        veredito = "VIÁVEL, MAS NÃO AGREGADOR — roda, porém não supera o K-Means; usar apenas como diagnóstico (ruído = outliers)."
    df.attrs["veredito"] = veredito
    return df


# ------------------------------------------------------------------------------
# 6. MODELO x CLUSTERS (ARQUÉTIPOS DE FRAUDE DO 03_Segmentacao)
# ------------------------------------------------------------------------------
def exportar_clusters_fraude_03(
    df_master: pd.DataFrame,
    df_fraude_treino: pd.DataFrame,
    preprocessador_cluster: Any,
    modelo_umap_10d: Any,
    kmeans_final: Any,
    mapa_arquetipos: Dict[int, str],
    caminho_cache: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Exporta oficialmente a partir do `03_Segmentacao` os rótulos de cluster ajustados no Treino
    e projeta as fraudes fora do Treino (Validação + OOT) usando EXATAMENTE o `preprocessador_cluster`,
    `umap_10d` e `kmeans_final` já ajustados na sessão do 03, garantindo paridade 100% determinística no 04.
    """
    df_cl_treino = pd.DataFrame({
        'TransactionID': df_fraude_treino['TransactionID'].values,
        'cluster': np.asarray(df_fraude_treino['cluster'], dtype=int),
        'base_cluster': 'Treino (ajuste)',
    })
    mask_fora = (df_master['isFraud'] == 1) & ~df_master['TransactionID'].isin(df_fraude_treino['TransactionID'])
    df_fora = df_master[mask_fora].sort_values('TransactionDT').reset_index(drop=True)
    partes = [df_cl_treino]
    if len(df_fora):
        X_fora = aplicar_preparo_clusterizacao(df_fora, preprocessador_cluster)
        rotulos_fora = kmeans_final.predict(modelo_umap_10d.transform(X_fora))
        partes.append(pd.DataFrame({
            'TransactionID': df_fora['TransactionID'].values,
            'cluster': rotulos_fora.astype(int),
            'base_cluster': 'Projetada (Val/OOT)',
        }))
    df_cl = pd.concat(partes, ignore_index=True)
    df_cl['arquetipo'] = df_cl['cluster'].map(mapa_arquetipos)
    if caminho_cache is not None:
        caminho = Path(caminho_cache)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        df_cl.to_parquet(caminho, index=False)
        print(f"[CLUSTERS 03->04] Exportados {len(df_cl_treino):,} rótulos de Treino + {len(df_fora):,} projetados (Val/OOT) em: {caminho}")
    return df_cl


def obter_rotulos_clusters_fraude(
    df_master: pd.DataFrame, meta: Optional[Dict[str, Any]], mapa_arquetipos: Dict[int, str],
    caminho_cache: Optional[Union[str, Path]] = None, k: int = 12, n_neighbors: int = 40, min_dist: float = 0.0,
    random_state: int = 42, forcar_recalculo: bool = False, train_ratio: float = 0.60,
) -> pd.DataFrame:
    """
    Rótulos de cluster (K-Means no UMAP 10D) e grupo/arquétipo para as fraudes do `df_master`.

    Carrega prioritariamente o cache oficial exportado pelo `03_Segmentacao` (`caminho_cache`),
    atualizando a coluna `arquetipo` com `mapa_arquetipos`. Caso o cache não exista, reproduz a
    esteira do 03_Segmentacao ajustando exclusivamente nas fraudes do Treino (`train_ratio`) e
    projetando Validação/OOT.
    """
    if caminho_cache is not None and Path(caminho_cache).exists() and not forcar_recalculo:
        df_cl = pd.read_parquet(caminho_cache)
        if mapa_arquetipos:
            novo_arq = df_cl['cluster'].map(mapa_arquetipos)
            if 'arquetipo' not in df_cl.columns or not df_cl['arquetipo'].equals(novo_arq):
                df_cl['arquetipo'] = novo_arq
                df_cl.to_parquet(caminho_cache, index=False)
        print(f"[CLUSTERS] Rótulos carregados do cache: {caminho_cache} ({len(df_cl):,} fraudes)")
        return df_cl

    from sklearn.cluster import KMeans
    print("[CLUSTERS] Cache não encontrado — reproduzindo a esteira do 03_Segmentacao (ajuste no Treino: UMAP 10D + K-Means)...")
    df_treino_ref = recortar_treino_temporal(df_master, train_ratio=train_ratio)
    df_fraude = df_treino_ref[df_treino_ref['isFraud'] == 1].sort_values('TransactionDT').reset_index(drop=True)
    X_tratado, _, _, preprocessador = preparar_dados_clusterizacao(
        df_fraude, meta=meta, max_ohe_cardinalidade=15, aplicar_log1p_assimetricas=True,
        limiar_skew_log=1.0)
    X_umap_10d, modelo_umap = projetar_umap(X_tratado, n_components=10, n_neighbors=n_neighbors, min_dist=min_dist,
                                            random_state=random_state)
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=random_state).fit(X_umap_10d)
    df_fraude = df_fraude.copy()
    df_fraude['cluster'] = kmeans.labels_.astype(int)
    return exportar_clusters_fraude_03(
        df_master=df_master,
        df_fraude_treino=df_fraude,
        preprocessador_cluster=preprocessador,
        modelo_umap_10d=modelo_umap,
        kmeans_final=kmeans,
        mapa_arquetipos=mapa_arquetipos,
        caminho_cache=caminho_cache,
    )
