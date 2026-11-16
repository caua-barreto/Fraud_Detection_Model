"""Diagnósticos e gráficos da EDA (01_EDA): tipos, nulos, cardinalidade, alvo e painéis de fraude.

Código migrado de `utils.py` (fachada de compatibilidade dos notebooks).
"""
from __future__ import annotations

from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
from src.visualization.theme import _mostrar_figura

try:
    from IPython.display import display
except ImportError:  # pragma: no cover
    display = print


def analisar_valores_nulos(
    df: pd.DataFrame,
    plotar: bool = True,
    top_n: Optional[int] = 25,
    figsize: Tuple[int, int] = (10, 6)
) -> pd.DataFrame:
    """
    Analisa os valores nulos do DataFrame, retornando uma tabela com a quantidade 
    e o percentual de dados faltantes por coluna. Opcionalmente, plota um gráfico.

    Parâmetros:
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado.
    plotar : bool, default=True
        Se True, gera um gráfico de barras horizontal com o percentual de nulos.
    top_n : int, optional, default=25
        Quantidade máxima de variáveis exibidas no gráfico (as com mais nulos).
        Se None, plota todas as colunas com nulos.
    figsize : tuple, default=(10, 6)
        Dimensões do gráfico gerado.

    Retorna:
    -------
    pd.DataFrame
        Tabela contendo colunas com nulos, ordenadas da maior para a menor quantidade.
    """
    total_nulos = df.isnull().sum()
    percentual_nulos = ((total_nulos / len(df)) * 100).round(2)
    
    df_nulos = pd.DataFrame({
        'Total Nulos': total_nulos, 
        'Percentual (%)': percentual_nulos,
        'Tipo de Dado': df.dtypes
    })
    
    df_nulos = df_nulos[df_nulos['Total Nulos'] > 0].sort_values(by='Total Nulos', ascending=False)
    
    if df_nulos.empty:
        print("Nenhum valor nulo encontrado no DataFrame!")
        return df_nulos

    if plotar:
        df_plot = df_nulos.head(top_n) if top_n is not None else df_nulos
        plt.figure(figsize=figsize)
        sns.set_theme(style="whitegrid")
        
        ax = sns.barplot(
            x='Percentual (%)', 
            y=df_plot.index, 
            data=df_plot, 
            color='#13579B', 
            edgecolor='black',
            alpha=0.8
        )
        
        for p in ax.patches:
            width = p.get_width()
            plt.text(
                width + 0.5, 
                p.get_y() + p.get_height() / 2.0 + 0.1, 
                f'{width:.1f}%', 
                ha="left", 
                va="center",
                fontsize=10,
                color='black'
            )
            
        sufixo_titulo = f" (Top {len(df_plot)})" if (top_n is not None and len(df_nulos) > top_n) else ""
        plt.title(f'Percentual de Valores Nulos por Variável{sufixo_titulo}', fontsize=14, fontweight='bold')
        plt.xlabel('Percentual (%)', fontsize=12)
        plt.ylabel('Variáveis', fontsize=12)
        plt.xlim(0, max(df_plot['Percentual (%)']) + 12)
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        _mostrar_figura()
        
    return df_nulos


analisar_nulos = analisar_valores_nulos


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str = 'isFraud',
    figsize: Tuple[int, int] = (14, 5.5)
) -> Tuple[pd.Series, pd.Series, float]:
    """
    Analisa e plota a distribuição da variável alvo em dois gráficos de colunas lado a lado:
    1. Quantidade absoluta do target rotulada (contagem + %)
    2. Proporção percentual (%) do target rotulada (%)

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame contendo a coluna alvo.
    target_col : str, default='isFraud'
        Nome da coluna target.
    figsize : tuple, default=(14, 5.5)
        Dimensões da figura.

    Retorna
    -------
    Tuple[pd.Series, pd.Series, float]
        (contagens, percentuais, ratio_classe_0_para_1)
    """
    target_counts = df[target_col].value_counts(dropna=False).sort_index()
    target_pct = (df[target_col].value_counts(dropna=False, normalize=True).sort_index() * 100).round(3)
    nulos_target = df[target_col].isnull().sum()
    ratio = target_counts.iloc[0] / target_counts.iloc[1] if len(target_counts) > 1 and target_counts.iloc[1] > 0 else np.nan

    print(f"--- Distribuição do Target ({target_col}) ---")
    print(f"  {target_col}=0 (Legítima / Adimplente): {target_counts.get(0, 0):>10,}  ({target_pct.get(0, 0.0):.3f}%)")
    print(f"  {target_col}=1 (Fraude / Default):      {target_counts.get(1, 0):>10,}  ({target_pct.get(1, 0.0):.3f}%)")
    print(f"  Valores nulos no target:               {nulos_target:>10,}")
    print(f"  Ratio (0:1):                           {ratio:>10.1f}:1\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor('#ffffff')

    cor_0 = "#8DB6E3"
    cor_1 = "#0B2545"
    cores = [cor_0, cor_1]
    bordas = ["#3E7CC4", "#06162B"]
    labels_x = [f"Não Fraude (0)", f"Fraude (1)"]

    # --- Painel 1: Quantidade Absoluta ---
    ax1.set_facecolor('#ffffff')
    bars1 = ax1.bar(
        [0, 1],
        target_counts.values,
        color=cores,
        edgecolor=bordas,
        linewidth=1.5,
        width=0.52
    )
    max_y1 = target_counts.max()
    for bar, count, pct in zip(bars1, target_counts.values, target_pct.values):
        texto = f"{count:,.0f}\n({pct:.2f}%)".replace(",", ".")
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max_y1 * 0.03),
            texto,
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold',
            color='#2C3E50'
        )
    ax1.set_title(
        f"Quantidade de Registros por Classe ({target_col})",
        fontsize=13,
        fontweight='bold',
        color='#2C3E50',
        pad=20
    )
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(labels_x, fontsize=11, fontweight='bold', color='#34495E')
    ax1.set_ylabel("Quantidade de Transações", fontsize=11, color='#34495E')
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x):,}".replace(",", ".")))
    ax1.set_ylim(0, max_y1 * 1.18)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)

    # --- Painel 2: Proporção Percentual (%) ---
    ax2.set_facecolor('#ffffff')
    bars2 = ax2.bar(
        [0, 1],
        target_pct.values,
        color=cores,
        edgecolor=bordas,
        linewidth=1.5,
        width=0.52
    )
    max_y2 = target_pct.max()
    for bar, pct in zip(bars2, target_pct.values):
        texto_pct = f"{pct:.2f}%".replace(".", ",")
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max_y2 * 0.03),
            texto_pct,
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold',
            color='#2C3E50'
        )
    ax2.set_title(
        f"Proporção Percentual (%) do Target ({target_col})",
        fontsize=13,
        fontweight='bold',
        color='#2C3E50',
        pad=20
    )
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(labels_x, fontsize=11, fontweight='bold', color='#34495E')
    ax2.set_ylabel("Percentual (%)", fontsize=11, color='#34495E')
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.0f}%"))
    ax2.set_ylim(0, max_y2 * 1.18)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    _mostrar_figura()

    return target_counts, target_pct, ratio


plot_distribuicao_target = plot_target_distribution


def analisar_tipos_colunas(df: pd.DataFrame, nome_df: str = "DataFrame") -> pd.DataFrame:
    """
    Conta e exibe a quantidade de colunas numéricas (inteiras e float) e categóricas
    (object e category) do DataFrame.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame a ser inspecionado.
    nome_df : str, default='DataFrame'
        Nome descritivo do DataFrame para exibição.

    Retorna
    -------
    pd.DataFrame
        Tabela resumo com a contagem de colunas por grupo de tipo.
    """
    int_cols = df.select_dtypes(include=["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]).columns
    float_cols = df.select_dtypes(include=["float16", "float32", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    print(f"--- Contagem de Colunas por Tipo ({nome_df}) ---")
    print(f"  Colunas numéricas inteiras : {len(int_cols)}")
    print(f"  Colunas numéricas float    : {len(float_cols)}")
    print(f"  Total de colunas numéricas : {len(int_cols) + len(float_cols)}")
    print(f"  Colunas categóricas        : {len(cat_cols)}")
    print(f"  Total geral de colunas     : {df.shape[1]}")

    resumo = pd.DataFrame({
        "Grupo de Tipo": ["Numéricas Inteiras", "Numéricas Float", "Categóricas (object/category)", "Total"],
        "Quantidade": [len(int_cols), len(float_cols), len(cat_cols), df.shape[1]]
    })
    return resumo


def analisar_nulos_por_faixa(df: pd.DataFrame, n_faixas: int = 20) -> pd.DataFrame:
    """
    Agrupa as colunas do DataFrame em faixas percentuais de valores nulos
    (por padrão 20 faixas de 5% em 5%, de 0% a 100%), exibindo a quantidade
    de colunas, o percentual sobre o total de colunas e exemplos de variáveis.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado.
    n_faixas : int, default=20
        Número de subdivisões entre 0% e 100% (20 = faixas de 5%).

    Retorna
    -------
    pd.DataFrame
        Tabela agrupada por faixa de nulos.
    """
    df_nulos = df.isnull().mean().mul(100).round(2).reset_index()
    df_nulos.columns = ["coluna", "proporcao_nulos"]

    bins = np.linspace(0, 100, n_faixas + 1)
    df_nulos["faixa_nulos"] = pd.cut(
        df_nulos["proporcao_nulos"],
        bins=bins,
        include_lowest=True
    )

    analise_bins = (
        df_nulos.groupby("faixa_nulos", observed=False)
        .agg(
            qtd_colunas=("coluna", "count"),
            exemplos=("coluna", lambda x: ", ".join(x.head(3)))
        )
        .reset_index()
    )

    analise_bins["pct_do_total_colunas"] = (
        (analise_bins["qtd_colunas"] / len(df_nulos)) * 100
    ).round(2)

    return analise_bins


def analisar_valores_unicos(
    df: pd.DataFrame,
    top_n: int = 35,
    ascending: bool = True
) -> pd.DataFrame:
    """
    Calcula a quantidade de valores únicos (incluindo NaN) por coluna do DataFrame.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado.
    top_n : int, default=35
        Quantidade de linhas a exibir no preview ordenado.
    ascending : bool, default=True
        Ordenação crescente (menor cardinalidade primeiro) se True.

    Retorna
    -------
    pd.DataFrame
        DataFrame completo com colunas ['coluna', 'valores_unicos'] ordenado.
    """
    df_unicos = df.nunique(dropna=False).reset_index()
    df_unicos.columns = ["coluna", "valores_unicos"]
    df_unicos = df_unicos.sort_values(by="valores_unicos", ascending=ascending).reset_index(drop=True)
    display(df_unicos.head(top_n))
    return df_unicos


def analisar_baixa_cardinalidade(
    df: pd.DataFrame,
    df_unicos: Optional[pd.DataFrame] = None,
    max_unicos: int = 5
) -> List[str]:
    """
    Identifica colunas com até `max_unicos` valores únicos (incluindo NaN) e
    imprime a distribuição percentual de cada uma.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado.
    df_unicos : pd.DataFrame, optional
        Resultado pré-calculado de `analisar_valores_unicos`. Se None, calcula internamente.
    max_unicos : int, default=5
        Limite máximo de valores distintos para considerar baixa cardinalidade.

    Retorna
    -------
    List[str]
        Lista de colunas com baixa cardinalidade.
    """
    if df_unicos is None:
        df_unicos = df.nunique(dropna=False).reset_index()
        df_unicos.columns = ["coluna", "valores_unicos"]

    colunas_baixa_card = df_unicos[df_unicos["valores_unicos"] <= max_unicos]["coluna"].tolist()
    colunas_baixa_card = [c for c in colunas_baixa_card if c in df.columns]

    print(f"Total de colunas com cardinalidade <= {max_unicos}: {len(colunas_baixa_card)}")
    for col in colunas_baixa_card:
        print(f"\n--- Distribuição da coluna: {col} ---")
        dist = df[col].value_counts(dropna=False, normalize=True) * 100
        print(dist.round(2).to_string())

    return colunas_baixa_card


def resumo_estatistico_numerico(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula as principais estatísticas descritivas para TODAS as variáveis
    numéricas do DataFrame, retornando uma matriz comparativa.
    """
    df_numerico = df.select_dtypes(include=[np.number])
    resultados = {}

    for coluna in df_numerico.columns:
        serie = df_numerico[coluna].dropna()
        if serie.empty:
            continue
        q1 = serie.quantile(0.25)
        q3 = serie.quantile(0.75)
        iqr = q3 - q1

        resultados[coluna] = [
            serie.mean(),
            serie.median(),
            serie.std(),
            serie.min(),
            q1,
            q3,
            iqr,
            serie.max(),
            serie.skew(),
            serie.kurt()
        ]

    metricas = [
        'Média',
        'Mediana',
        'Desvio Padrão',
        'Mínimo',
        '1º Quartil (Q1 - 25%)',
        '3º Quartil (Q3 - 75%)',
        'Intervalo Interquartil (IQR)',
        'Máximo',
        'Assimetria (Skewness)',
        'Curtose (Kurtosis)'
    ]

    df_resumo = pd.DataFrame(resultados, index=metricas).T
    return df_resumo.round(2)


# ==============================================================================
# PAINÉIS ANALÍTICOS DA EDA (COMPORTAMENTO DA FRAUDE)
# ==============================================================================
def plot_painel_fraude_bivariado(
    df: pd.DataFrame,
    col_target: str = 'isFraud',
    col_hora: str = 'tempo_hora',
    col_valor: str = 'TransactionAmt',
    col_canal: str = 'ProductCD',
    cor_fraude: str = '#0B2545',
    cor_legitima: str = '#48CAE4',
    figsize: tuple = (18, 5),
):
    """
    Painel bivariado com 3 gráficos lado a lado:
      1) Taxa de fraude (%) por hora do dia, com linha da média global;
      2) Densidade de log1p(valor) por classe (fraude x legítima);
      3) Taxa de fraude (%) por canal de produto, ordenada do maior para o menor risco.

    Retorna
    -------
    fig : matplotlib.figure.Figure
    """
    media_global = df[col_target].mean() * 100
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. Taxa de Fraude por Hora do Dia
    hora_stats = df.groupby(col_hora)[col_target].mean() * 100
    axes[0].plot(hora_stats.index, hora_stats.values, marker='o', color=cor_fraude, linewidth=2)
    axes[0].axhline(media_global, color='gray', linestyle='--', label=f'Média Global ({media_global:.1f}%)')
    axes[0].set_title('Taxa de Fraude por Hora do Dia (%)', fontweight='bold')
    axes[0].set_xlabel('Hora do Dia (0h - 23h)')
    axes[0].set_ylabel('Taxa de Fraude (%)')
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    # 2. Distribuição de Valor (Log) por Classe
    sns.kdeplot(
        data=df, x=np.log1p(df[col_valor]),
        hue=col_target, common_norm=False, fill=True,
        palette={0: cor_legitima, 1: cor_fraude}, ax=axes[1],
    )
    axes[1].set_title(f'Distribuição de log1p({col_valor})', fontweight='bold')
    axes[1].set_xlabel(f'log1p({col_valor})')
    axes[1].set_ylabel('Densidade')
    axes[1].legend(title=col_target, labels=['Fraude (1)', 'Legítima (0)'])
    axes[1].grid(alpha=0.2)

    # 3. Taxa de Fraude por Canal
    canal_stats = (df.groupby(col_canal)[col_target].mean() * 100).sort_values(ascending=False)
    bars = axes[2].bar(canal_stats.index.astype(str), canal_stats.values, color=cor_fraude, alpha=0.85)
    axes[2].axhline(media_global, color='gray', linestyle='--')
    axes[2].set_title(f'Taxa de Fraude por Canal ({col_canal})', fontweight='bold')
    axes[2].set_xlabel(col_canal)
    axes[2].set_ylabel('Taxa de Fraude (%)')
    for bar in bars:
        h = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width() / 2, h + 0.2, f'{h:.1f}%', ha='center', fontsize=9)
    axes[2].grid(axis='y', alpha=0.2)

    plt.tight_layout()
    _mostrar_figura()
    return fig


def plot_evolucao_temporal_fraude(
    df: pd.DataFrame,
    col_target: str = 'isFraud',
    col_dia: str = 'tempo_dia_corrido',
    cor_fraude: str = '#0B2545',
    cor_volume: str = '#8DB6E3',
    figsize: tuple = (18, 4.5),
):
    """
    Evolução diária (Concept Drift): barras de volume de transações + linha da taxa
    de fraude (%) em eixo secundário, com a média global como referência pontilhada.

    Retorna
    -------
    evolucao_dia : pd.DataFrame
        Tabela com colunas [col_dia, 'volume', 'taxa_fraude'].
    """
    evolucao_dia = df.groupby(col_dia)[col_target].agg(
        volume='count',
        taxa_fraude=lambda x: x.mean() * 100,
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.bar(evolucao_dia[col_dia], evolucao_dia['volume'], color=cor_volume, alpha=0.35, label='Volume de Transações')
    ax1.set_xlabel('Dias Corridos (TransactionDT)')
    ax1.set_ylabel('Volume Diário de Transações', color='#4a4a4a')
    ax1.tick_params(axis='y', labelcolor='#4a4a4a')

    ax2 = ax1.twinx()
    ax2.plot(evolucao_dia[col_dia], evolucao_dia['taxa_fraude'], color=cor_fraude, linewidth=1.8, label='Taxa de Fraude (%)')
    ax2.axhline(df[col_target].mean() * 100, color=cor_fraude, linestyle=':', alpha=0.6)
    ax2.set_ylabel('Taxa de Fraude Diária (%)', color=cor_fraude)
    ax2.tick_params(axis='y', labelcolor=cor_fraude)

    plt.title('Evolução Temporal: Volume Transacional vs. Taxa de Fraude (Concept Drift)', fontweight='bold')
    ax1.grid(alpha=0.2)
    plt.tight_layout()
    _mostrar_figura()
    return evolucao_dia
