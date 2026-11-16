"""Gráficos de clusterização e arquétipos de fraude (03_Segmentacao).

Código migrado de `utils.py` (fachada de compatibilidade dos notebooks).
"""
from __future__ import annotations

from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

from src.visualization.theme import (  # noqa: F401
    COR_RUIDO_CLUSTER,
    PALETA_ARQUETIPOS_PADRAO,
    PALETA_AZUL,
    _COR_TEXTO_PRIMARIO,
    _COR_TEXTO_SECUNDARIO,
    _salvar_fig,
    _salvar_figura_opcional,
    _titulo_ax,
)
from src.data.preprocess import (  # noqa: F401
    reverter_escala_log1p,
)
from src.models.threshold import (  # noqa: F401
    threshold_dinamico,
)
from src.segmentation.fraud_clusters import (  # noqa: F401
    _resolver_colunas_perfil_fraude,
    _serie_diaria_arquetipos,
)


def plot_pca_variancia_acumulada(
    pca_model: Any,
    limiar_referencia: float = 0.70,
    nome_arquivo: Optional[str] = None,
    save_dir: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plota a variância explicada individual e acumulada dos componentes do PCA."""
    ratios = pca_model.explained_variance_ratio_
    acum = np.cumsum(ratios)
    comps = np.arange(1, len(ratios) + 1)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(comps, ratios * 100, alpha=0.65, color="#8DB6E3", label="Variância Individual (%)")
    ax.plot(comps, acum * 100, marker="o", linewidth=2.2, color="#0B2545", label="Variância Acumulada (%)")
    ax.axhline(limiar_referencia * 100, color="#0096C7", linestyle="--", linewidth=1.3, label=f"Referência {limiar_referencia*100:.0f}%")

    for c, v in zip(comps, acum * 100):
        ax.annotate(f"{v:.1f}%", (c, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, fontweight="bold", color="#0B2545")

    ax.set_title("PCA — Variância Explicada Acumulada", fontweight="bold", fontsize=12)
    ax.set_xlabel("Componente Principal")
    ax.set_ylabel("Variância Explicada (%)")
    ax.set_xticks(comps)
    ax.set_ylim(0, min(105, max(100, acum.max() * 100 + 12)))
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    sns.despine(ax=ax)
    plt.tight_layout()
    _salvar_figura_opcional(fig, nome_arquivo=nome_arquivo, save_dir=save_dir)
    return fig


def plot_elbow_comparativo(
    resultados_dict: Dict[str, Union[Dict[int, float], pd.DataFrame]],
    nome_arquivo: Optional[str] = None,
    save_dir: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plota curvas de Elbow lado a lado para múltiplos espaços (ex: PCA 10D vs UMAP 10D)."""
    n_espacos = len(resultados_dict)
    fig, axes = plt.subplots(1, n_espacos, figsize=(6.2 * n_espacos, 4.6))
    if n_espacos == 1:
        axes = [axes]

    cores = ["#0B2545", "#0096C7", "#2E86C1", "#48CAE4"]
    for idx, (titulo, dados) in enumerate(resultados_dict.items()):
        ax = axes[idx]
        if isinstance(dados, pd.DataFrame):
            ks = dados["K"].values
            inercias = dados["Inercia"].values
        else:
            ks = list(dados.keys())
            inercias = list(dados.values())
        ax.plot(ks, inercias, marker="o", linewidth=2.2, color=cores[idx % len(cores)])
        ax.set_title(f"Método do Cotovelo — {titulo}", fontweight="bold", fontsize=11)
        ax.set_xlabel("Número de Clusters (K)")
        ax.set_ylabel("Inércia (SSE)")
        ax.set_xticks(ks)
        ax.grid(alpha=0.25)
        sns.despine(ax=ax)

    plt.suptitle("K-Means — Comparação de Curvas Elbow", fontweight="bold", fontsize=13, y=1.03)
    plt.tight_layout()
    _salvar_figura_opcional(fig, nome_arquivo=nome_arquivo, save_dir=save_dir)
    return fig


def plot_metricas_validacao(
    df_metricas: pd.DataFrame,
    titulo_espaco: str = "",
    nome_arquivo: Optional[str] = None,
    save_dir: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plota Silhouette (↑), Davies-Bouldin (↓) e Calinski-Harabasz (↑) ao longo de K (adaptado do Projeto-07)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ks = df_metricas["K"].values
    configs = [
        ("Silhouette", "Silhouette Score (↑ melhor)", "#13579B", True),
        ("Davies_Bouldin", "Davies-Bouldin (↓ melhor)", "#0096C7", False),
        ("Calinski_Harabasz", "Calinski-Harabasz (↑ melhor)", "#0B2545", True),
    ]

    for ax, (col, titulo, cor, maior_melhor) in zip(axes, configs):
        vals = df_metricas[col].values
        ax.plot(ks, vals, marker="o", linewidth=2.0, color=cor)
        idx_melhor = int(np.argmax(vals) if maior_melhor else np.argmin(vals))
        k_best = ks[idx_melhor]
        v_best = vals[idx_melhor]
        ax.scatter([k_best], [v_best], s=110, color="#48CAE4", zorder=5, edgecolor="#0B2545", linewidth=1.5, label=f"Melhor K={k_best} ({v_best:.3f})")
        ax.set_title(titulo, fontweight="bold", fontsize=10.5)
        ax.set_xlabel("K")
        ax.set_xticks(ks)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8.5)
        sns.despine(ax=ax)

    sufixo = f" — {titulo_espaco}" if titulo_espaco else ""
    plt.suptitle(f"Métricas Internas de Validação (K-Means){sufixo}", fontweight="bold", fontsize=13, y=1.03)
    plt.tight_layout()
    _salvar_figura_opcional(fig, nome_arquivo=nome_arquivo, save_dir=save_dir)
    return fig


def plot_k_distance_dbscan(
    X_espaco: np.ndarray,
    k: int = 15,
    titulo_espaco: str = "UMAP 10D",
    nome_arquivo: Optional[str] = None,
    save_dir: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Calcula e plota o gráfico de k-distância (k-NN) para estimar o `eps` do DBSCAN."""
    from sklearn.neighbors import NearestNeighbors

    vizinhos = NearestNeighbors(n_neighbors=k)
    vizinhos.fit(X_espaco)
    distancias, _ = vizinhos.kneighbors(X_espaco)
    k_dists = np.sort(distancias[:, k - 1])

    p90, p95, p99 = np.percentile(k_dists, [90, 95, 99])
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.plot(k_dists, linewidth=1.8, color="#0B2545", label=f"Distância ao {k}º vizinho")
    ax.axhline(p90, color="#48CAE4", linestyle="--", linewidth=1.2, label=f"P90 = {p90:.3f}")
    ax.axhline(p95, color="#0096C7", linestyle="--", linewidth=1.2, label=f"P95 = {p95:.3f}")
    ax.axhline(p99, color="#13579B", linestyle="--", linewidth=1.2, label=f"P99 = {p99:.3f}")

    ax.set_title(f"Curva k-Distance (k={k}) — Estimativa de eps ({titulo_espaco})", fontweight="bold", fontsize=11.5)
    ax.set_xlabel("Amostras Ordenadas por Distância")
    ax.set_ylabel(f"Distância até o {k}º Vizinho")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    sns.despine(ax=ax)
    plt.tight_layout()
    _salvar_figura_opcional(fig, nome_arquivo=nome_arquivo, save_dir=save_dir)
    return fig, k_dists


def plot_metricas_dbscan(
    df_res: pd.DataFrame,
    titulo_espaco: str = "",
    nome_arquivo: Optional[str] = None,
    save_dir: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plota Silhouette, % de Ruído e N_Clusters na grade do DBSCAN (adaptado do Projeto-07)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    cores = ["#0B2545", "#0096C7", "#2E86C1", "#48CAE4", "#8DB6E3"]

    for idx, ms in enumerate(sorted(df_res["min_samples"].unique())):
        sub = df_res[df_res["min_samples"] == ms].sort_values("eps")
        cor = cores[idx % len(cores)]
        lbl = f"min_samples={ms}"
        axes[0].plot(sub["eps"], sub["Silhouette"], marker="o", linewidth=1.8, color=cor, label=lbl)
        axes[1].plot(sub["eps"], sub["%_Ruido"], marker="s", linewidth=1.8, color=cor, label=lbl)
        axes[2].plot(sub["eps"], sub["N_Clusters"], marker="^", linewidth=1.8, color=cor, label=lbl)

    axes[0].set_title("Silhouette (sem ruído -1) ↑", fontweight="bold")
    axes[1].set_title("% Pontos classificados como Ruído (-1)", fontweight="bold")
    axes[2].set_title("Número de Clusters Descobertos", fontweight="bold")
    for ax in axes:
        ax.set_xlabel("eps")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8.5)
        sns.despine(ax=ax)

    sufixo = f" — {titulo_espaco}" if titulo_espaco else ""
    plt.suptitle(f"Sensibilidade DBSCAN{sufixo}", fontweight="bold", fontsize=13, y=1.03)
    plt.tight_layout()
    _salvar_figura_opcional(fig, nome_arquivo=nome_arquivo, save_dir=save_dir)
    return fig


def plot_metricas_hdbscan(
    df_res: pd.DataFrame,
    titulo_espaco: str = "",
    nome_arquivo: Optional[str] = None,
    save_dir: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plota DBCV, Silhouette, % Ruído e N_Clusters na grade do HDBSCAN (adaptado do Projeto-07)."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    cores = ["#0B2545", "#0096C7", "#2E86C1", "#48CAE4"]

    for idx, ms in enumerate(sorted(df_res["min_samples"].unique())):
        sub = df_res[df_res["min_samples"] == ms].sort_values("min_cluster_size")
        cor = cores[idx % len(cores)]
        lbl = f"min_samples={ms}"
        axes[0].plot(sub["min_cluster_size"], sub["DBCV"], marker="o", linewidth=1.8, color=cor, label=lbl)
        axes[1].plot(sub["min_cluster_size"], sub["Silhouette"], marker="o", linewidth=1.8, color=cor, label=lbl)
        axes[2].plot(sub["min_cluster_size"], sub["%_Ruido"], marker="s", linewidth=1.8, color=cor, label=lbl)
        axes[3].plot(sub["min_cluster_size"], sub["N_Clusters"], marker="^", linewidth=1.8, color=cor, label=lbl)

    axes[0].set_title("DBCV (relative_validity_) ↑", fontweight="bold")
    axes[1].set_title("Silhouette (sem ruído -1) ↑", fontweight="bold")
    axes[2].set_title("% Ruído (-1) ↓", fontweight="bold")
    axes[3].set_title("Nº de Clusters", fontweight="bold")
    for ax in axes:
        ax.set_xlabel("min_cluster_size")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        sns.despine(ax=ax)

    sufixo = f" — {titulo_espaco}" if titulo_espaco else ""
    plt.suptitle(f"Sensibilidade HDBSCAN{sufixo}", fontweight="bold", fontsize=13, y=1.03)
    plt.tight_layout()
    _salvar_figura_opcional(fig, nome_arquivo=nome_arquivo, save_dir=save_dir)
    return fig


def plot_distribuicao_clusters(
    labels: Union[np.ndarray, pd.Series],
    titulo: str = "Distribuição de Fraudes por Cluster",
    nome_arquivo: Optional[str] = None,
    save_dir: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plota volume absoluto e % por cluster, destacando ruído (-1) em cinza (adaptado do Projeto-07)."""
    contagem = pd.Series(labels).value_counts().sort_index()
    total = contagem.sum()

    fig, ax = plt.subplots(figsize=(max(7.5, len(contagem) * 0.9), 4.6))
    cores = ["#97A3B0" if idx == -1 else "#13579B" for idx in contagem.index]
    nomes = ["Ruído (-1)" if idx == -1 else f"Cluster {idx}" for idx in contagem.index]

    bars = ax.bar(nomes, contagem.values, color=cores, edgecolor="black", alpha=0.88)
    for bar, val in zip(bars, contagem.values):
        pct = val / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.01,
            f"{val:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
        )

    ax.set_title(titulo, fontweight="bold", fontsize=12)
    ax.set_ylabel("Número de Transações Fraudulentas")
    ax.set_ylim(0, contagem.max() * 1.18)
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20 if len(nomes) > 6 else 0)
    sns.despine(ax=ax)
    plt.tight_layout()
    _salvar_figura_opcional(fig, nome_arquivo=nome_arquivo, save_dir=save_dir)
    return fig


def plot_clusters_2d_plotly(
    X_2d: np.ndarray,
    labels: Union[np.ndarray, pd.Series],
    titulo: str = "Projeção UMAP 2D das Fraudes por Cluster",
    hover_df: Optional[pd.DataFrame] = None,
    sample_plot: Optional[int] = 15000,
    random_state: int = 42,
    nome_arquivo: Optional[str] = None,
    save_dir: Optional[Union[str, Path]] = None,
) -> Any:
    """Gera scatter interativo Plotly em 2D colorindo por cluster e destacando ruído (-1) (adaptado do Projeto-07)."""
    try:
        import plotly.express as px
    except ImportError as exc:
        raise ImportError("Pacote 'plotly' não encontrado. Instale com `pip install plotly`.") from exc

    X_arr = np.asarray(X_2d)
    if X_arr.ndim != 2 or X_arr.shape[1] < 2:
        raise ValueError(f"`X_2d` deve ter shape (n, >=2); recebido {X_arr.shape}.")
    labels_arr = np.asarray(labels)
    if len(labels_arr) != len(X_arr):
        raise ValueError(f"Tamanhos incompatíveis: X_2d={len(X_arr)} vs labels={len(labels_arr)}.")

    df_plot = pd.DataFrame({
        "Dim_1": X_arr[:, 0],
        "Dim_2": X_arr[:, 1],
        "Cluster": [
            "Ruído (-1)" if (lbl == -1 or str(lbl) == "-1" or str(lbl) == "Ruído (-1)") else f"Cluster {lbl}"
            for lbl in labels_arr
        ],
    })

    hover_cols: List[str] = []
    if hover_df is not None:
        if len(hover_df) != len(X_arr):
            raise ValueError(f"Tamanhos incompatíveis: X_2d={len(X_arr)} vs hover_df={len(hover_df)}.")
        for col in ["TransactionAmt", "ProductCD", "card4", "device_brand", "P_emaildomain"]:
            if col in hover_df.columns:
                vals = hover_df[col].reset_index(drop=True)
                if col == "TransactionAmt" and pd.api.types.is_numeric_dtype(vals):
                    vals_num = pd.to_numeric(vals, errors="coerce").astype(float)
                    if not vals_num.dropna().empty and float(vals_num.dropna().max()) <= 20:
                        df_plot["Ticket_USD"] = np.round(np.expm1(vals_num), 2)
                        hover_cols.append("Ticket_USD")
                    else:
                        df_plot[col] = np.round(vals_num, 2)
                        hover_cols.append(col)
                else:
                    df_plot[col] = vals.astype(object).where(vals.notna(), "N/A").astype(str)
                    hover_cols.append(col)

    if sample_plot is not None and len(df_plot) > sample_plot:
        df_plot = df_plot.sample(n=sample_plot, random_state=random_state)

    def _chave_ordem_cluster(nome: str) -> Tuple[int, int, str]:
        if nome == "Ruído (-1)":
            return (2, 0, nome)
        suf = nome.replace("Cluster ", "", 1)
        return (0, int(suf), nome) if suf.lstrip("-").isdigit() else (1, 0, nome)

    ordem_clusters = sorted(df_plot["Cluster"].unique(), key=_chave_ordem_cluster)
    color_map = {"Ruído (-1)": COR_RUIDO_CLUSTER}
    sequencia_azul = ["#0B2545", "#13579B", "#0077B6", "#0096C7", "#00B4D8", "#2E86C1", "#48CAE4", "#5DADE2", "#8DB6E3", "#1D3557"]

    fig = px.scatter(
        df_plot,
        x="Dim_1",
        y="Dim_2",
        color="Cluster",
        labels={"Dim_1": "UMAP 1", "Dim_2": "UMAP 2"},
        category_orders={"Cluster": ordem_clusters},
        color_discrete_map=color_map,
        color_discrete_sequence=sequencia_azul,
        hover_data=hover_cols if hover_cols else None,
        title=titulo,
        opacity=0.68,
        render_mode="svg",
        template="plotly_white",
    )
    fig.update_traces(marker=dict(size=4.5))
    fig.update_layout(height=580, legend_title_text="Cluster", legend=dict(itemsizing="constant"))
    _salvar_figura_opcional(fig, nome_arquivo=nome_arquivo, save_dir=save_dir)
    return fig


def plot_violino_perfil_clusters(
    df_fraude: pd.DataFrame,
    col_cluster: str = "cluster_campeao",
    cols_numericas: Optional[List[str]] = None,
    ignorar_ruido: bool = True,
    nome_arquivo: Optional[str] = None,
    save_dir: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plota distribuição (boxplot/violino) das principais variáveis por cluster (adaptado do Projeto-07)."""
    df = df_fraude.copy()
    if ignorar_ruido:
        df = df[df[col_cluster] != -1].copy()

    if cols_numericas is None:
        candidatas = ["TransactionAmt", "tr_null_count", "id_null_count", "C1", "C14", "TransactionAmt_to_mean_card1"]
        cols_numericas = [c for c in candidatas if c in df.columns][:4]

    n_cols = len(cols_numericas)
    if n_cols == 0:
        raise ValueError("Nenhuma variável numérica válida encontrada para o gráfico de perfil.")

    df["_cluster_label"] = df[col_cluster].apply(lambda x: "Ruído (-1)" if x == -1 else f"C{x}")
    ordem = sorted(df["_cluster_label"].unique(), key=lambda s: (s == "Ruído (-1)", s))

    fig, axes = plt.subplots(1, n_cols, figsize=(4.8 * n_cols, 4.6))
    if n_cols == 1:
        axes = [axes]

    for ax, col in zip(axes, cols_numericas):
        sns.boxplot(data=df, x="_cluster_label", y=col, order=ordem, hue="_cluster_label", legend=False, palette="Blues", showfliers=False, ax=ax)
        ax.set_title(col, fontweight="bold", fontsize=11)
        ax.set_xlabel("Cluster")
        ax.grid(axis="y", alpha=0.25)
        plt.setp(ax.get_xticklabels(), rotation=20)
        sns.despine(ax=ax)

    plt.suptitle("Assinatura Comportamental dos Clusters (Distribuição Sem Outliers Extremos)", fontweight="bold", fontsize=13, y=1.03)
    plt.tight_layout()
    _salvar_figura_opcional(fig, nome_arquivo=nome_arquivo, save_dir=save_dir)
    return fig


def plot_optuna_clusterizacao(
    df_trials: pd.DataFrame,
    metrica_alvo: str = "DBCV",
    titulo: str = "Histórico de Otimização Optuna",
    nome_arquivo: Optional[str] = None,
    save_dir: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plota evolução da métrica-alvo nos trials válidos e trade-off entre % de ruído e qualidade."""
    validos = df_trials[df_trials["objetivo"] > -1.0].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5))

    if not validos.empty:
        axes[0].plot(validos["trial"], validos["objetivo"], marker="o", linestyle="-", alpha=0.75, color="#2E86C1", label="Trial válido")
        axes[0].plot(validos["trial"], validos["objetivo"].cummax(), linestyle="--", linewidth=2.0, color="#0B2545", label="Melhor acumulado")
        sc = axes[1].scatter(validos["%_Ruido"], validos["objetivo"], c=validos["N_Clusters"], cmap="PuBu", s=65, edgecolor="#0B2545", alpha=0.85)
        plt.colorbar(sc, ax=axes[1], label="Nº de Clusters")
    else:
        axes[0].text(0.5, 0.5, "Nenhum trial viável encontrado", ha="center", va="center", transform=axes[0].transAxes)
        axes[1].text(0.5, 0.5, "Nenhum trial viável encontrado", ha="center", va="center", transform=axes[1].transAxes)

    axes[0].set_title(f"Evolução Optuna — {metrica_alvo} ↑", fontweight="bold")
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel(metrica_alvo)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8.5)

    axes[1].set_title(f"Trade-off % Ruído vs {metrica_alvo}", fontweight="bold")
    axes[1].set_xlabel("% Ruído (-1)")
    axes[1].set_ylabel(metrica_alvo)
    axes[1].grid(alpha=0.25)

    for ax in axes:
        sns.despine(ax=ax)

    plt.suptitle(titulo, fontweight="bold", fontsize=12.5, y=1.03)
    plt.tight_layout()
    _salvar_figura_opcional(fig, nome_arquivo=nome_arquivo, save_dir=save_dir)
    return fig


def plot_heatmap_zscore_clusters(
    df_fraude: pd.DataFrame,
    col_cluster: str = "cluster",
    features_continuas: Optional[List[str]] = None,
    titulo: str = "Assinatura Numérica dos Clusters de Fraude (Modus Operandi)",
    ignorar_ruido: bool = True,
    nome_arquivo: Optional[str] = None,
    save_dir: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plota heatmap de Z-Score das médias contínuas por cluster (reconverte log1p se necessário)."""
    from matplotlib.colors import LinearSegmentedColormap

    df_work = df_fraude.copy()
    if ignorar_ruido:
        df_work = df_work[df_work[col_cluster] != -1].copy()

    mapa_cols = _resolver_colunas_perfil_fraude(df_work)
    cols_contagem = [
        c for c in mapa_cols["nulos"] + mapa_cols["vel"] + mapa_cols["delta"]
        if pd.api.types.is_numeric_dtype(df_work[c])
    ]
    df_work, _ = reverter_escala_log1p(
        df_work,
        cols_monetarias=["TransactionAmt"],
        cols_contagem=cols_contagem,
        verbose=False,
    )

    # Sem ancoragem: deltas D*/delta_D* usados na escala original
    padrao_feats = (
        (["TransactionAmt"] if "TransactionAmt" in df_work.columns else [])
        + mapa_cols["nulos"]
        + mapa_cols["vel"]
        + mapa_cols["delta"]
        + mapa_cols["extras"]
    )
    candidatas = features_continuas if features_continuas is not None else padrao_feats
    cols_validas = [
        c for c in candidatas
        if c in df_work.columns and pd.api.types.is_numeric_dtype(df_work[c])
    ]
    if not cols_validas:
        raise ValueError("Nenhuma feature numérica válida encontrada para o heatmap de Z-Score.")

    medias = df_work.groupby(col_cluster, sort=True)[cols_validas].mean()
    stds = medias.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    zscore_clusters = ((medias - medias.mean()) / stds).fillna(0.0)

    cmap_corp = LinearSegmentedColormap.from_list("corp_blues", ["#F5F9FD", "#8DB6E3", "#0096C7", "#0B2545"])
    altura = max(6.0, 0.55 * len(cols_validas))
    largura = max(10.0, 1.15 * len(zscore_clusters))

    fig, ax = plt.subplots(figsize=(largura, altura))
    sns.heatmap(
        zscore_clusters.T,
        cmap=cmap_corp,
        center=0,
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        cbar_kws={"label": "Z-Score (Desvios da Média Global das Fraudes)"},
        ax=ax,
    )
    ax.set_title(titulo, fontsize=13.5, fontweight="bold", pad=15, loc="left")
    ax.set_xlabel("ID do Cluster", fontsize=11, fontweight="bold")
    ax.set_ylabel("Features Numéricas / Comportamentais", fontsize=11, fontweight="bold")
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    _salvar_figura_opcional(fig, nome_arquivo=nome_arquivo, save_dir=save_dir)
    return fig


def plot_volume_vs_prejuizo_arquetipos(
    df_tabela: pd.DataFrame,
    col_grupo: str = "Arquetipo",
    col_volume: str = "%_Fraudes",
    col_prejuizo: str = "%_Prejuizo_Total",
    titulo: str = "Participação em quantidade vs. prejuízo (US$) por arquétipo",
    nome_arquivo: Optional[str] = None,
    save_dir: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Barras lado a lado: share de quantidade (%) vs. share de prejuízo em US$ (%) por arquétipo.

    As cores codificam a **métrica** (cinza claro = volume, grafite = prejuízo), não o arquétipo,
    para não conflitar com a paleta categórica dos demais gráficos. O rótulo do eixo X traz a
    alavancagem (prejuízo% / volume%): > 1× indica impacto financeiro desproporcional ao volume.
    """
    df = df_tabela[[col_grupo, col_volume, col_prejuizo]].copy()
    df["_alav"] = df[col_prejuizo] / df[col_volume]

    cor_volume, cor_prejuizo = "#8DB6E3", "#0B2545"
    x = np.arange(len(df))
    largura = 0.38
    folga = 0.02  # espaço de 2% entre barras adjacentes

    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    b_vol = ax.bar(x - largura / 2 - folga, df[col_volume], width=largura, color=cor_volume, label="Volume (% das fraudes)")
    b_prej = ax.bar(x + largura / 2 + folga, df[col_prejuizo], width=largura, color=cor_prejuizo, label="Prejuízo (% do US$ total)")

    teto = float(max(df[col_volume].max(), df[col_prejuizo].max()))
    for barras in (b_vol, b_prej):
        for bar in barras:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + teto * 0.015, f"{h:.1f}%",
                    ha="center", va="bottom", fontsize=9.5, color=_COR_TEXTO_PRIMARIO, fontweight="bold")

    rotulos = [f"{g}\nalavancagem {a:.2f}×" for g, a in zip(df[col_grupo], df["_alav"])]
    ax.set_xticks(x)
    ax.set_xticklabels(rotulos, fontsize=10, color=_COR_TEXTO_PRIMARIO)
    ax.set_ylabel("Participação (%)", color=_COR_TEXTO_SECUNDARIO)
    ax.set_ylim(0, teto * 1.18)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.04), ncol=2, fontsize=9.5)
    ax.set_title(titulo, fontweight="bold", fontsize=12.5, loc="left", pad=34)
    sns.despine(ax=ax)
    plt.tight_layout()
    _salvar_figura_opcional(fig, nome_arquivo=nome_arquivo, save_dir=save_dir)
    return fig


def plot_violino_arquetipos(
    df_fraude: pd.DataFrame,
    col_grupo: str = "arquetipo",
    cols: Optional[Dict[str, str]] = None,
    ordem: Optional[List[str]] = None,
    paleta: Optional[Dict[str, str]] = None,
    quantil_corte: float = 0.99,
    titulo: str = "Assinatura comportamental dos 4 arquétipos (escala real, cauda > p99 removida)",
    nome_arquivo: Optional[str] = None,
    save_dir: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Violinos (mediana e quartis tracejados) das variáveis-chave por arquétipo, em escala real.

    `cols` mapeia coluna -> rótulo do eixo Y. Valores acima do quantil global `quantil_corte`
    de cada variável são excluídos **apenas do gráfico** para que a cauda extrema não esmague
    o eixo. Arquétipos sem nenhum valor (ex.: `id_null_count` 100% NaN) recebem uma nota.
    """
    if cols is None:
        cols = {
            "TransactionAmt": "Ticket (US$)",
            "vel_C2": "vel_C2 — velocidade/recorrência (contagem)",
            "cont_null_id": "cont_null_id (nulos de identidade)",
            "cont_null_tr": "cont_null_tr (nulos transacionais)",
        }
    aliases = {
        "C2": "vel_C2", "vel_C2": "C2",
        "id_null_count": "cont_null_id", "cont_null_id": "id_null_count",
        "tr_null_count": "cont_null_tr", "cont_null_tr": "tr_null_count",
        "D1": "delta_D1", "delta_D1": "D1",
    }
    cols_resolvidas: Dict[str, str] = {}
    for c, r in cols.items():
        if c in df_fraude.columns:
            cols_resolvidas[c] = r
        elif aliases.get(c) in df_fraude.columns:
            cols_resolvidas[aliases[c]] = r
    cols = cols_resolvidas
    if not cols:
        raise ValueError("Nenhuma das colunas solicitadas está presente em df_fraude.")

    df_real, _ = reverter_escala_log1p(
        df_fraude[[col_grupo] + list(cols)],
        cols_monetarias=[c for c in cols if c == "TransactionAmt"],
        cols_contagem=[c for c in cols if c != "TransactionAmt"],
        verbose=False,
    )
    ordem = ordem or sorted(df_real[col_grupo].dropna().unique())
    paleta = paleta or dict(zip(ordem, PALETA_ARQUETIPOS_PADRAO))

    n = len(cols)
    n_lin, n_col = (2, 2) if n == 4 else (1, n)
    fig, axes = plt.subplots(n_lin, n_col, figsize=(7.2 * n_col, 4.6 * n_lin))
    axes = np.atleast_1d(axes).ravel()

    for ax, (col, rotulo) in zip(axes, cols.items()):
        serie = df_real[col]
        corte = float(serie.quantile(quantil_corte)) if serie.notna().any() else np.nan
        df_plot = df_real.loc[serie.notna() & (serie <= corte), [col_grupo, col]]
        presentes = [g for g in ordem if int((df_plot[col_grupo] == g).sum()) >= 5]
        df_plot = df_plot[df_plot[col_grupo].isin(presentes)]

        if presentes:
            sns.violinplot(
                data=df_plot, x=col_grupo, y=col, order=ordem, hue=col_grupo, hue_order=ordem,
                palette=paleta, inner="quart", cut=0, density_norm="width", linewidth=1.0,
                saturation=1.0, legend=False, ax=ax,
            )
            for coll in ax.collections:
                coll.set_alpha(0.85)
            # Mediana como ponto branco com borda escura (legível sobre qualquer cor)
            medianas = df_plot.groupby(col_grupo, observed=True)[col].median()
            for i, g in enumerate(ordem):
                if g in medianas.index and pd.notna(medianas[g]):
                    ax.scatter(i, medianas[g], s=36, color="white", edgecolor=_COR_TEXTO_PRIMARIO, linewidth=1.2, zorder=5)

        y_min, y_max = ax.get_ylim()
        for i, g in enumerate(ordem):
            if g not in presentes:
                ax.text(i, y_min + (y_max - y_min) * 0.5, "sem registro\nde identidade\n(~100% NaN)",
                        ha="center", va="center", fontsize=8.5, color=_COR_TEXTO_SECUNDARIO, style="italic")

        ax.set_title(f"{rotulo}  ·  exibido ≤ p{int(quantil_corte * 100)} ({corte:,.0f})",
                     fontweight="bold", fontsize=10.5, loc="left", color=_COR_TEXTO_PRIMARIO)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks(range(len(ordem)))
        ax.set_xticklabels(ordem, fontsize=9.5)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        sns.despine(ax=ax)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(titulo, fontweight="bold", fontsize=13, x=0.01, ha="left", y=0.985)
    fig.text(0.01, 0.942, "Ponto branco = mediana · linhas tracejadas = quartis Q1/Q3 · largura = densidade relativa no grupo",
             fontsize=9, color=_COR_TEXTO_SECUNDARIO, ha="left")
    plt.tight_layout(rect=(0, 0, 1, 0.90))
    _salvar_figura_opcional(fig, nome_arquivo=nome_arquivo, save_dir=save_dir)
    return fig


def plot_evolucao_temporal_arquetipos(
    df_fraude: pd.DataFrame,
    col_grupo: str = "arquetipo",
    col_dt: str = "TransactionDT",
    ordem: Optional[List[str]] = None,
    paleta: Optional[Dict[str, str]] = None,
    janela_mm: int = 7,
    dias_por_bloco: int = 7,
    titulo: str = "Dinâmica temporal dos arquétipos ao longo da janela de fraudes",
    nome_arquivo: Optional[str] = None,
    save_dir: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Small multiples do volume diário (linha fina + média móvel) e participação semanal no prejuízo.

    - Painéis 1..4: um por arquétipo, eixo Y próprio (rajadas ficam visíveis mesmo em grupos menores);
      o pico diário é marcado com um ponto.
    - Painel final: área 100% empilhada da participação de cada arquétipo no prejuízo (US$) por bloco
      de `dias_por_bloco` dias — mostra quem ganha/perde relevância financeira no tempo.
    """
    import matplotlib.gridspec as gridspec

    ordem = ordem or sorted(df_fraude[col_grupo].dropna().unique())
    paleta = paleta or dict(zip(ordem, PALETA_ARQUETIPOS_PADRAO))
    volume, prejuizo = _serie_diaria_arquetipos(df_fraude, col_grupo, col_dt, ordem)

    n = len(ordem)
    fig = plt.figure(figsize=(15, 1.75 * n + 3.6))
    gs = gridspec.GridSpec(n + 1, 1, height_ratios=[1] * n + [1.9], hspace=0.55, figure=fig)
    ax_ref = None

    for i, g in enumerate(ordem):
        ax = fig.add_subplot(gs[i], sharex=ax_ref) if ax_ref is not None else fig.add_subplot(gs[i])
        ax_ref = ax_ref or ax
        s = volume[g]
        mm = s.rolling(janela_mm, center=True, min_periods=1).mean()
        ax.plot(s.index, s.values, color=paleta[g], linewidth=0.8, alpha=0.35)
        ax.plot(mm.index, mm.values, color=paleta[g], linewidth=2.0)
        dia_pico = int(s.idxmax())
        ax.scatter([dia_pico], [s.max()], s=40, color=paleta[g], edgecolor="white", linewidth=1.5, zorder=5)
        ax.set_title(
            f"{g}  ·  pico {int(s.max())} fraudes no dia {dia_pico}  ·  mediana {s.median():.0f}/dia",
            loc="left", fontsize=10, fontweight="bold", color=_COR_TEXTO_PRIMARIO,
        )
        ax.set_ylim(0, s.max() * 1.15 if s.max() > 0 else 1)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(3, integer=True))
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="x", labelbottom=False, length=0)
        sns.despine(ax=ax)

    # Participação semanal no prejuízo (US$), 100% empilhada
    ax_sh = fig.add_subplot(gs[n], sharex=ax_ref)
    bloco = (prejuizo.index // dias_por_bloco)
    prej_bloco = prejuizo.groupby(bloco).sum()
    share = prej_bloco.div(prej_bloco.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
    x_bloco = share.index.values * dias_por_bloco + dias_por_bloco / 2
    ax_sh.stackplot(
        x_bloco, *[share[g].values for g in ordem], labels=ordem,
        colors=[paleta[g] for g in ordem], alpha=0.9, edgecolor="white", linewidth=1.0,
    )
    ax_sh.set_ylim(0, 100)
    ax_sh.set_xlim(volume.index.min(), volume.index.max())
    ax_sh.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax_sh.set_ylabel("% do prejuízo", color=_COR_TEXTO_SECUNDARIO)
    ax_sh.set_xlabel("Dias desde a primeira fraude da base (TransactionDT / 86.400)", color=_COR_TEXTO_SECUNDARIO)
    ax_sh.set_title(f"Participação no prejuízo (US$) por bloco de {dias_por_bloco} dias",
                    loc="left", fontsize=10, fontweight="bold", color=_COR_TEXTO_PRIMARIO)
    ax_sh.legend(ncol=n, frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, -0.28))
    sns.despine(ax=ax_sh)

    fig.suptitle(titulo, fontweight="bold", fontsize=13, x=0.01, ha="left", y=0.995)
    fig.text(0.01, 0.965, f"Volume diário de fraudes: linha fina = diário · linha grossa = média móvel {janela_mm} dias · ponto = pico",
             fontsize=9, color=_COR_TEXTO_SECUNDARIO, ha="left")
    fig.subplots_adjust(top=0.92, bottom=0.12, left=0.06, right=0.98)
    _salvar_figura_opcional(fig, nome_arquivo=nome_arquivo, save_dir=save_dir)
    return fig


def plot_arquetipos_2d_plotly(
    X_2d: np.ndarray,
    arquetipos: Union[np.ndarray, pd.Series],
    ordem: Optional[List[str]] = None,
    paleta: Optional[Dict[str, str]] = None,
    titulo: str = "Topologia UMAP 2D colorida pelos 4 arquétipos",
    hover_df: Optional[pd.DataFrame] = None,
    sample_plot: Optional[int] = 15000,
    random_state: int = 42,
    nome_arquivo: Optional[str] = None,
    save_dir: Optional[Union[str, Path]] = None,
) -> Any:
    """Reaproveita `plot_clusters_2d_plotly` e aplica nomes/cores fixas dos arquétipos."""
    arq = pd.Series(arquetipos).astype(str).reset_index(drop=True)
    ordem = ordem or sorted(arq.unique())
    paleta = paleta or {g: PALETA_ARQUETIPOS_PADRAO[i % len(PALETA_ARQUETIPOS_PADRAO)] for i, g in enumerate(ordem)}

    fig = plot_clusters_2d_plotly(
        X_2d, arq.values, titulo=titulo, hover_df=hover_df,
        sample_plot=sample_plot, random_state=random_state,
    )
    ordem_idx = {g: i for i, g in enumerate(ordem)}

    def _ajustar(trace):
        nome = str(trace.name).replace("Cluster ", "", 1)
        hover = (trace.hovertemplate or "").replace("Cluster=Cluster ", "Arquétipo=").replace("Cluster=", "Arquétipo=")
        trace.update(
            name=nome,
            legendgroup=nome,
            legendrank=ordem_idx.get(nome, 99),
            marker_color=paleta.get(nome, COR_RUIDO_CLUSTER),
            marker_size=4.5,
            hovertemplate=hover,
        )

    fig.for_each_trace(_ajustar)
    fig.update_layout(legend_title_text="Arquétipo", legend=dict(itemsizing="constant"))
    _salvar_figura_opcional(fig, nome_arquivo=nome_arquivo, save_dir=save_dir)
    return fig


def plot_crosstab_hdbscan_arquetipos(
    crosstab_pct: pd.DataFrame,
    ordem: Optional[List[str]] = None,
    titulo: str = "Para qual arquétipo cada cluster HDBSCAN aponta? (% da linha)",
    nome_arquivo: Optional[str] = None,
    save_dir: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Heatmap sequencial (um matiz) da composição de cada cluster HDBSCAN pelos 4 arquétipos."""
    cols = ordem or [c for c in crosstab_pct.columns if c not in ("Arquetipo_Dominante", "Pureza_%", "N")]
    dados = crosstab_pct[cols].astype(float)
    rotulos_linha = [f"{idx}  (n={int(n):,})" for idx, n in zip(dados.index, crosstab_pct["N"])]

    fig, ax = plt.subplots(figsize=(1.9 * len(cols) + 3.5, 0.42 * len(dados) + 1.8))
    sns.heatmap(
        dados, cmap="Blues", vmin=0, vmax=100, annot=True, fmt=".0f", linewidths=1.5, linecolor="white",
        cbar_kws={"label": "% do cluster HDBSCAN", "shrink": 0.8}, annot_kws={"fontsize": 9}, ax=ax,
    )
    ax.set_yticklabels(rotulos_linha, rotation=0, fontsize=9)
    ax.set_xticklabels(cols, rotation=0, fontsize=9.5)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(length=0)
    ax.set_title(titulo, fontweight="bold", fontsize=12, loc="left", pad=12)
    plt.tight_layout()
    _salvar_figura_opcional(fig, nome_arquivo=nome_arquivo, save_dir=save_dir)
    return fig


def plot_curvas_exponenciais_por_grupamento(
    res_exp_k4: Dict[str, Any],
    regra_dinamica_global: Dict[str, Any],
    amt_max: float = 600.0,
    figsize: Tuple[float, float] = (15, 5.4),
    filepath: Optional[Union[str, Path]] = None,
) -> None:
    """
    Painel duplo das curvas exponenciais por grupamento (K=4):
      1. Curva T_g(amt) vs. Valor da Transação (US$) para cada grupo frente ao Exponencial Global
      2. Amplitude [t_min_g, t_max_g] de cada grupo comparada à banda do Exponencial Global.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    cores_g = [PALETA_AZUL['marinho'], PALETA_AZUL['principal'], PALETA_AZUL['ciano'], PALETA_AZUL['claro']]
    df_exp = res_exp_k4['df_exp_grupos']
    mapa_p = res_exp_k4['mapa_params_exp']
    eixo_amt = np.linspace(0.0, amt_max, 400)

    for idx, row in df_exp.iterrows():
        gid = int(row['Grupo_ID'])
        p_g = mapa_p[gid]
        cor = cores_g[idx % len(cores_g)]
        curva_g = threshold_dinamico(eixo_amt, p_g['t_min'], p_g['t_max'], p_g['k']) * 100
        rot_curto = row['Grupamento'].split(' (')[0]
        ax1.plot(
            eixo_amt, curva_g, color=cor, lw=2.3,
            label=f"{rot_curto}: [{row['t_min_Grupo_%']:.2f}% → {row['t_max_Grupo_%']:.2f}%] (k={row['k_Grupo']:.4f})",
        )

    curva_glob = threshold_dinamico(
        eixo_amt, regra_dinamica_global['t_min'], regra_dinamica_global['t_max'], regra_dinamica_global['k']
    ) * 100
    ax1.plot(
        eixo_amt, curva_glob, color=PALETA_AZUL['texto'], lw=2.2, ls='--',
        label=f"Exponencial Global: [{regra_dinamica_global['t_min']*100:.2f}% → {regra_dinamica_global['t_max']*100:.2f}%]",
    )
    ax1.set(xlim=(0, amt_max), xlabel='Valor da Transação (US$)', ylabel='Threshold dinâmico T_g(amt) (%)')
    ax1.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='white', fontsize=8.3)
    _titulo_ax(ax1, 'Curvas Dinâmicas Exponenciais por Grupamento (K=4)',
               'Decaimento do ponto de corte conforme o valor da transação dentro de cada grupo')

    y_pos = np.arange(len(df_exp))
    rotulos_curtos = [r.split(' · ')[0] + ')' if ' · ' in r else r for r in df_exp['Grupamento']]
    for idx, row in df_exp.iterrows():
        cor = cores_g[idx % len(cores_g)]
        xmin, xmax = float(row['t_min_Grupo_%']), float(row['t_max_Grupo_%'])
        ax2.hlines(idx, xmin, xmax, color=cor, lw=6, alpha=0.88)
        ax2.scatter([xmin, xmax], [idx, idx], color=cor, s=65, zorder=4, edgecolor='white', lw=1.2)
        ax2.text(xmax + 0.18, idx, f"{xmin:.2f}% – {xmax:.2f}%", va='center', fontsize=8.5,
                 fontweight='bold', color=PALETA_AZUL['texto'])

    t_min_glob_pct = float(regra_dinamica_global['t_min'] * 100)
    t_max_glob_pct = float(regra_dinamica_global['t_max'] * 100)
    ax2.axvspan(t_min_glob_pct, t_max_glob_pct, color=PALETA_AZUL['ciano_claro'], alpha=0.22, zorder=0,
                label=f'Faixa Exponencial Global [{t_min_glob_pct:.2f}% – {t_max_glob_pct:.2f}%]')
    ax2.set(
        yticks=y_pos, yticklabels=rotulos_curtos,
        xlabel='Intervalo [t_min (alto ticket) — t_max (micro-compra)] (%)',
        xlim=(0, max(float(df_exp['t_max_Grupo_%'].max()), t_max_glob_pct) * 1.28),
    )
    ax2.invert_yaxis()
    ax2.grid(axis='y', visible=False)
    ax2.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='white', fontsize=8.5)
    _titulo_ax(ax2, 'Amplitude [t_min, t_max] por Grupamento vs. Global',
               'Comparação da banda de variação de corte aprendida em cada macrogrupo')

    plt.tight_layout()
    _salvar_fig(fig, filepath)


def plot_thresholds_por_grupamento(
    res_otim_k4: Dict[str, Any],
    threshold_fixo_global: float,
    regra_dinamica: Dict[str, Any],
    figsize: Tuple[float, float] = (15, 5.4),
    filepath: Optional[Union[str, Path]] = None,
) -> None:
    """
    Painel duplo da otimização por grupamento (K=4):
      1. Curvas de Lucro Líquido no Treino (OOF) por threshold para cada grupamento
      2. Comparação dos thresholds ótimos por grupamento vs. Fixo Global e faixa do Dinâmico Exponencial.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    cores_g = [PALETA_AZUL['marinho'], PALETA_AZUL['principal'], PALETA_AZUL['ciano'], PALETA_AZUL['claro']]
    df_th = res_otim_k4['df_thresholds']
    curvas = res_otim_k4['curvas_por_grupo']

    for idx, row in df_th.iterrows():
        rot = row['Grupamento']
        cor = cores_g[idx % len(cores_g)]
        cv = curvas[rot]
        ax1.plot(cv['Threshold'] * 100, cv['Lucro'] / 1e3, color=cor, lw=2.2, label=rot.split(' (')[0] + f" (t*={row['Threshold_Optuna_%']:.2f}%)")
        th_ot = row['Threshold_Optuna_%'] / 100.0
        luc_ot = row['Lucro_OOF_Grupamento_USD'] / 1e3
        ax1.scatter([th_ot * 100], [luc_ot], color=cor, s=55, zorder=5, edgecolor='white', lw=1.2)

    ax1.axvline(threshold_fixo_global * 100, color=PALETA_AZUL['texto_sec'], ls='--', lw=1.4,
                label=f'Fixo Global ({threshold_fixo_global*100:.2f}%)')
    ax1.set(xlim=(0.5, 15.0), xlabel='Threshold de probabilidade calibrada (%)', ylabel='Lucro Líquido OOF (US$ mil)')
    ax1.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='white', fontsize=8.5)
    _titulo_ax(ax1, 'Curva de Lucro OOF por Grupamento (K=4)', 'Cada grupo atinge seu ótimo em um ponto distinto')

    y_pos = np.arange(len(df_th))
    rotulos_curtos = [r.split(' · ')[0] + ')' if ' · ' in r else r for r in df_th['Grupamento']]
    bars = ax2.barh(y_pos, df_th['Threshold_Optuna_%'], color=cores_g[:len(df_th)], height=0.55, alpha=0.92)
    ax2.bar_label(bars, fmt='%.2f%%', padding=5, fontsize=9, fontweight='bold', color=PALETA_AZUL['texto'])

    t_min_pct = float(regra_dinamica['t_min'] * 100)
    t_max_pct = float(regra_dinamica['t_max'] * 100)
    ax2.axvspan(t_min_pct, t_max_pct, color=PALETA_AZUL['ciano_claro'], alpha=0.22, zorder=0,
                label=f'Faixa Exponencial [{t_min_pct:.2f}% – {t_max_pct:.2f}%]')
    ax2.axvline(threshold_fixo_global * 100, color=PALETA_AZUL['texto_sec'], ls='--', lw=1.5,
                label=f'Fixo Global ({threshold_fixo_global*100:.2f}%)')

    ax2.set(yticks=y_pos, yticklabels=rotulos_curtos, xlabel='Threshold ótimo de bloqueio (%)',
            xlim=(0, max(float(df_th['Threshold_Optuna_%'].max()), t_max_pct) * 1.28))
    ax2.invert_yaxis()
    ax2.grid(axis='y', visible=False)
    ax2.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='white', fontsize=8.5)
    _titulo_ax(ax2, 'Thresholds por Grupamento vs. Faixa Exponencial',
               'Comparação dos cortes fixos por cluster com a banda dinâmica por valor')

    plt.tight_layout()
    _salvar_fig(fig, filepath)


def plot_benchmark_grupamento_vs_exponencial(
    df_comp: pd.DataFrame,
    figsize: Tuple[float, float] = (16.5, 5.2),
    filepath: Optional[Union[str, Path]] = None,
) -> None:
    """
    Compara o Lucro Líquido (US$ mil) na Validação (In-Time) e o Trade-off Aceitação × Recall
    entre Fixo Global, Fixo por Grupamento (K=4 e K=10), Dinâmico Exponencial Global e Exponencial por Grupamento.
    """
    df_plot = df_comp[df_comp['Política'] != '1. Sem Modelo'].copy()
    base_ref = 'Validação' if 'Validação' in df_plot['Base'].values else df_plot['Base'].iloc[-1]
    sub_ref = df_plot[df_plot['Base'] == base_ref].reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    paleta_pol = [
        PALETA_AZUL['cinza'],
        PALETA_AZUL['claro'],
        PALETA_AZUL['ciano_claro'],
        PALETA_AZUL['marinho'],
        PALETA_AZUL['principal'],
        PALETA_AZUL['medio'],
        PALETA_AZUL['ciano'],
    ]
    marcadores = ['s', '^', 'v', '*', 'D', 'P', 'X']

    y_pos = np.arange(len(sub_ref))
    nomes_curtos = [p.split('. ', 1)[-1] for p in sub_ref['Política']]
    vals_val = sub_ref['Lucro Líquido (US$)'].values / 1e3
    cores_barras = [paleta_pol[i % len(paleta_pol)] for i in range(len(sub_ref))]

    bars = ax1.barh(y_pos, vals_val, color=cores_barras, height=0.58, alpha=0.93)
    for bar, (_, r) in zip(bars, sub_ref.iterrows()):
        delta = r['Δ vs. Exponencial Global (US$)']
        sinal = f"({delta:+,.0f} US\\$)" if abs(delta) >= 1 else "(Referência)"   # \$ evita o modo matemático do matplotlib
        ax1.text(
            bar.get_width() + 0.35,
            bar.get_y() + bar.get_height() / 2,
            f"\\$ {r['Lucro Líquido (US$)']/1e3:.2f}k  {sinal}",
            va='center',
            fontsize=8.3,
            fontweight='bold',
            color=PALETA_AZUL['texto'],
        )

    xmax = max(float(vals_val.max()), 1.0)
    ax1.set(yticks=y_pos, yticklabels=nomes_curtos, xlabel='Lucro Líquido na Validação (US$ mil)', xlim=(0, xmax * 1.45))
    ax1.invert_yaxis()
    ax1.grid(axis='y', visible=False)
    _titulo_ax(ax1, f'Lucro Líquido por Estratégia ({base_ref})',
               'Avaliação fora do treino na partição de Validação (In-Time)')

    for i, r in sub_ref.iterrows():
        cor = paleta_pol[i % len(paleta_pol)]
        nome_curto = r['Política'].split('. ', 1)[-1]
        eh_atual = 'Exponencial Global (Atual)' in nome_curto
        mk = '*' if eh_atual else marcadores[i % len(marcadores)]
        sz = 280 if eh_atual else 125
        z = 6 if eh_atual else 4
        ax2.scatter(
            r['Aceitação Geral (%)'],
            r['Recall (%)'],
            s=sz,
            marker=mk,
            color=cor,
            edgecolor='white',
            lw=1.3,
            alpha=0.95,
            zorder=z,
            label=f"{nome_curto} (\\$ {r['Lucro Líquido (US$)']/1e3:.1f}k)",
        )
        if eh_atual:
            ax2.annotate(
                f"Atual (\\$ {r['Lucro Líquido (US$)']/1e3:.1f}k)",
                (r['Aceitação Geral (%)'], r['Recall (%)']),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=8.6,
                fontweight='bold',
                color=cor,
                arrowprops=dict(arrowstyle='->', color=cor, lw=1.0, shrinkA=2, shrinkB=4),
            )

    x_vals = sub_ref['Aceitação Geral (%)'].values
    y_vals = sub_ref['Recall (%)'].values
    dx = max(float(x_vals.max() - x_vals.min()), 0.5)
    dy = max(float(y_vals.max() - y_vals.min()), 0.5)
    ax2.set(
        xlim=(float(x_vals.min()) - 0.16 * dx, float(x_vals.max()) + 0.18 * dx),
        ylim=(float(y_vals.min()) - 0.16 * dy, float(y_vals.max()) + 0.18 * dy),
        xlabel='Aceitação Geral de Pedidos (%) → menor atrito',
        ylabel='Recall de Fraudes (%) → maior captura',
    )
    ax2.legend(
        loc='center left',
        bbox_to_anchor=(1.03, 0.5),
        ncol=1,
        frameon=True,
        facecolor='white',
        edgecolor=PALETA_AZUL['cinza_claro'],
        title='Políticas (Lucro na Validação)',
        title_fontsize=8.8,
        fontsize=8.3,
        labelspacing=0.65,
        markerscale=0.95,
    )
    _titulo_ax(ax2, f'Trade-off Aceitação × Recall ({base_ref})',
               'Canto superior direito = mais pedidos aprovados com maior captura de fraude')

    plt.tight_layout(w_pad=2.5)
    _salvar_fig(fig, filepath)
