import gc
import os
import sys
import math
import json
import warnings
from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple, Any, Iterable

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

try:
    from IPython.display import display
except ImportError:
    display = print

# Scikit-Learn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold, cross_val_score, cross_val_predict, train_test_split
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# Modelos e explicabilidade (com fallback gracioso)
try:
    import joblib
except ImportError:
    joblib = None

try:
    import shap
except ImportError:
    shap = None

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant
except ImportError:
    variance_inflation_factor = None
    add_constant = None

def histograma_simples(
    df: pd.DataFrame, 
    coluna: str = 'price', 
    bins: Union[int, str] = 'sturges',
    titulo: str = None,
    xlabel: str = None,
    ylabel: str = 'Frequência',
    color: str = 'steelblue',
    figsize: tuple = (10, 6)
):
    """
    Gera um gráfico combinado com Boxplot (topo) e Histograma com KDE (base),
    destacando as linhas de Média, Mediana e Moda.

    Parâmetros:
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    coluna : str, default='price'
        Nome da coluna numérica a ser analisada.
    bins : int ou str, default='sturges'
        Número de compartimentos do histograma ou método de cálculo (ex: 30, 'auto', 'sturges').
    titulo : str, optional
        Título principal do gráfico.
    xlabel : str, optional
        Rótulo do eixo X. Se None, usa o nome da coluna.
    ylabel : str, default='Frequência'
        Rótulo do eixo Y.
    color : str, default='steelblue'
        Cor base para o boxplot e o histograma.
    figsize : tuple, default=(10, 6)
        Dimensões da figura (largura, altura).
    """
    # 1. Extração dos dados e medidas de tendência central
    dados = df[coluna].dropna()
    media = dados.mean()
    mediana = dados.median()
    
    # Tratamento caso a coluna não tenha moda definida
    modas = dados.mode()
    moda = modas[0] if not modas.empty else None

    # 2. Configuração dos subplots compartilhando o eixo X
    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, 
        sharex=True, 
        gridspec_kw={"height_ratios": (1, 4)}, 
        figsize=figsize
    )
    
    # Adiciona o título principal, se fornecido
    if titulo:
        fig.suptitle(titulo, fontsize=14, fontweight='bold')

    # 3. Boxplot Superior (usando a cor personalizada)
    sns.boxplot(x=dados, ax=ax_box, color=color, fliersize=4)
    ax_box.axvline(media, color='#0B2545', linestyle='--', linewidth=1.5)
    ax_box.axvline(mediana, color='#0096C7', linestyle='--', linewidth=1.5)
    if moda is not None:
        ax_box.axvline(moda, color='#3E7CC4', linestyle='--', linewidth=1.5)
    ax_box.set(xlabel='')
    ax_box.grid(True, linestyle='--', alpha=0.7)

    # 4. Histograma com KDE Inferior (usando bins e cor personalizados)
    sns.histplot(
        x=dados, 
        bins=bins, 
        kde=True, 
        ax=ax_hist, 
        color=color, 
        edgecolor='black', 
        alpha=0.6
    )

    # Adição das linhas verticais estatísticas no histograma
    ax_hist.axvline(media, color='#0B2545', linestyle='--', linewidth=2, label=f'Média: {media:.2f}')
    ax_hist.axvline(mediana, color='#0096C7', linestyle='--', linewidth=2, label=f'Mediana: {mediana:.2f}')
    if moda is not None:
        ax_hist.axvline(moda, color='#3E7CC4', linestyle='--', linewidth=2, label=f'Moda: {moda:.2f}')

    # 5. Formatação e Layout (usando os rótulos personalizados)
    ax_hist.legend(frameon=True, fontsize=10)
    
    # Define o xlabel como o fornecido, ou usa o nome da coluna se for None
    rotulo_x = xlabel if xlabel is not None else coluna
    ax_hist.set_xlabel(rotulo_x, fontsize=12)
    
    ax_hist.set_ylabel(ylabel, fontsize=12)
    ax_hist.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    
    # Ajusta o topo do gráfico para não sobrepor o título, caso exista
    if titulo:
        plt.subplots_adjust(top=0.92)
        
    plt.show()


def resumo_estatistico(df: pd.DataFrame, coluna: str = 'price', monetario: bool = True) -> pd.DataFrame:
    """
    Calcula as principais estatísticas descritivas de uma variável numérica,
    incluindo quartis, IQR, assimetria e curtose.

    Parâmetros:
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    coluna : str, default='price'
        Nome da coluna numérica a ser analisada.
    monetario : bool, default=True
        Se True, formata os valores numéricos com o prefixo '$'.
        Se False, formata com duas casas decimais simples.

    Retorna:
    -------
    pd.DataFrame
        Tabela com as métricas e seus respectivos valores.
    """
    serie = df[coluna].dropna()
    
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    
    fmt = lambda v: f"${v:,.2f}" if monetario else f"{v:,.2f}"
    
    metricas = {
        'Métrica': [
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
        ],
        'Valor': [
            fmt(serie.mean()),
            fmt(serie.median()),
            fmt(serie.std()),
            fmt(serie.min()),
            fmt(q1),
            fmt(q3),
            fmt(iqr),
            fmt(serie.max()),
            f"{serie.skew():.2f}",
            f"{serie.kurt():.2f}"
        ]
    }
    
    return pd.DataFrame(metricas)

def plot_distribuicao(
    df: pd.DataFrame, 
    coluna: str = 'price', 
    transformacao: str = 'original', 
    cor: str = None, 
    bins: int = 50, 
    figsize: tuple = (10, 6)
) -> pd.DataFrame:
    """
    Plota a distribuição de uma variável com KDE e ajuste normal teórico,
    e retorna as principais métricas estatísticas da distribuição.

    Parâmetros:
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    coluna : str, default='price'
        Coluna numérica a ser analisada.
    transformacao : str, default='original'
        Tipo de transformação: 'original', 'sqrt', 'log', 'log1p', 'cbrt'.
    cor : str, optional
        Cor do histograma. Aceita nomes (ex: 'royalblue') ou Hex (ex: '#0096C7').
    bins : int, default=50
        Número de bins do histograma.
    figsize : tuple, default=(10, 6)
        Dimensões da figura (largura, altura).
        
    Retorna:
    -------
    pd.DataFrame
        DataFrame contendo Média, Mediana, Desvio Padrão, Assimetria e Curtose.
    """
    serie = df[coluna].dropna()

    padrao_cores = {
        'original': 'royalblue',
        'sqrt': 'darkorange',
        'log': 'forestgreen',
        'log1p': 'seagreen',
        'cbrt': 'purple'
    }

    t = (transformacao or 'original').lower()

    # Definição dos dados transformados e cores
    if t in ('original', 'none', 'nenhuma'):
        dados = serie
        label_trans = 'Original'
        cor_final = cor if cor else padrao_cores['original']
    elif t == 'sqrt':
        dados = np.sqrt(serie)
        label_trans = 'Raiz Quadrada (Sqrt)'
        cor_final = cor if cor else padrao_cores['sqrt']
    elif t == 'log':
        dados = np.log(serie)
        label_trans = 'Logaritmo Natural (log)'
        cor_final = cor if cor else padrao_cores['log']
    elif t == 'log1p':
        dados = np.log1p(serie)
        label_trans = 'Logaritmo Natural (log1p)'
        cor_final = cor if cor else padrao_cores['log1p']
    elif t == 'cbrt':
        dados = np.cbrt(serie)
        label_trans = 'Raiz Cúbica (cbrt)'
        cor_final = cor if cor else padrao_cores['cbrt']
    else:
        raise ValueError(
            f"Transformação '{transformacao}' inválida. Opções: 'original', 'sqrt', 'log', 'log1p', 'cbrt'."
        )

    plt.figure(figsize=figsize)

    # 1. Histograma com densidade KDE (O parâmetro color já aceita Hex nativamente)
    sns.histplot(
        dados, 
        kde=True, 
        stat="density", 
        color=cor_final, 
        bins=bins, 
        edgecolor='black', 
        alpha=0.6, 
        label='Densidade Real (KDE)'
    )

    # 2. Ajuste da Normal Teórica
    mu, std = stats.norm.fit(dados)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 200)
    p = stats.norm.pdf(x, mu, std)

    plt.plot(
        x, p, 'r--', linewidth=2, 
        label=f'Ajuste Normal Teórico\n($\\mu$={mu:,.2f}, $\\sigma$={std:,.2f})'
    )

    # 3. Formatação Visual
    plt.title(f'Distribuição de {coluna} ({label_trans}) com Ajuste Normal', fontsize=13, fontweight='bold')
    plt.xlabel(f'{coluna} ({label_trans})', fontsize=11)
    plt.ylabel('Densidade', fontsize=11)
    plt.legend(frameon=True, fontsize=10)
    plt.grid(axis='y', alpha=0.5, linestyle='--')

    plt.tight_layout()
    plt.show()

    # 4. Cálculo das Estatísticas para Retorno
    df_stats = pd.DataFrame({
        'Transformação': [label_trans],
        'Média': [dados.mean()],
        'Mediana': [dados.median()],
        'Desvio Padrão': [dados.std()],
        'Assimetria (Skew)': [dados.skew()],
        'Curtose (Kurt)': [dados.kurt()]
    })
    
    # Formatação para visualização limpa
    for col in ['Média', 'Mediana', 'Desvio Padrão', 'Assimetria (Skew)', 'Curtose (Kurt)']:
        df_stats[col] = df_stats[col].apply(lambda val: f"{val:.4f}")
        
    return df_stats

def plot_boxplot_outliers(
    df: pd.DataFrame, 
    coluna: str = 'price', 
    monetario: bool = True, 
    cor: str = '#2E86C1', 
    figsize: tuple = (12, 5)
) -> dict:
    """
    Gera um boxplot horizontal destacando os limites de Tukey e outliers.

    Parâmetros:
    ----------
    df : pd.DataFrame
        DataFrame com os dados.
    coluna : str, default='price'
        Nome da coluna numérica.
    monetario : bool, default=True
        Se True, formata o eixo X com '$' e notação 'K'.
    cor : str, default='#2E86C1'
        Cor de preenchimento do boxplot.
    figsize : tuple, default=(12, 5)
        Dimensões da figura.

    Retorna:
    -------
    dict
        Dicionário com estatísticas dos limites e contagem de outliers.
    """
    serie = df[coluna].dropna()
    
    # 1. Cálculos estatísticos (Regra IQR de Tukey)
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    limite_superior = q3 + 1.5 * iqr
    limite_inferior = q1 - 1.5 * iqr
    
    outliers_sup = serie[serie > limite_superior]
    outliers_inf = serie[serie < limite_inferior]
    total_outliers = len(outliers_sup) + len(outliers_inf)

    # 2. Plotagem
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(
        x=serie, 
        color=cor, 
        width=0.4, 
        linewidth=1.5,
        ax=ax,
        flierprops={
            'marker': 'o',
            'markerfacecolor': 'white', 
            'markeredgecolor': '#4A4A4A',
            'alpha': 0.5,
            'markersize': 5
        }
    )

    # 3. Linha e anotação do limite superior
    ax.axvline(limite_superior, color='#2B2B2B', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Deslocamento dinâmico baseado na amplitude dos dados
    offset = (serie.max() - serie.min()) * 0.015
    texto_corte = f'Limite Superior\n${limite_superior:,.2f}' if monetario else f'Limite Superior\n{limite_superior:,.2f}'
    
    ax.text(
        limite_superior + offset, -0.28, 
        texto_corte, 
        color='#2B2B2B', fontsize=10, va='center'
    )

    # 4. Formatação de eixos
    if monetario:
        formatter = ticker.FuncFormatter(lambda x, pos: f'${x/1000:,.1f}K' if abs(x) >= 1000 else f'${x:,.0f}')
        ax.xaxis.set_major_formatter(formatter)

    ax.set_title(f'Distribuição de {coluna} e Detecção de Outliers', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel(f'{coluna} ($)' if monetario else coluna, fontsize=12, labelpad=10)
    ax.set_yticks([])
    sns.despine(left=True)

    plt.tight_layout()
    plt.show()

    # 5. Saída de diagnóstico
    print(f"--- Diagnóstico de Outliers: {coluna} ---")
    print(f"Limite Superior (Q3 + 1.5*IQR): {limite_superior:,.2f}")
    print(f"Limite Inferior (Q1 - 1.5*IQR): {limite_inferior:,.2f}")
    print(f"Outliers Superiores: {len(outliers_sup)} registros")
    print(f"Outliers Inferiores: {len(outliers_inf)} registros")
    print(f"Total de Outliers: {total_outliers} ({(total_outliers / len(serie)) * 100:.2f}%)")

    return {
        'limite_inferior': limite_inferior,
        'limite_superior': limite_superior,
        'qtd_outliers_sup': len(outliers_sup),
        'qtd_outliers_inf': len(outliers_inf),
        'total_outliers': total_outliers
    }

def plot_correlacao_alvo(
    df: pd.DataFrame, 
    alvo: str = 'price',
    top_n: int = 10,
    metodo: str = 'pearson',
    cor: str = 'Blues',
    figsize: tuple = (4, 8)
) -> pd.DataFrame:
    """
    Gera um heatmap de correlação focado APENAS na variável alvo.
    Retorna um DataFrame com TODAS as features ordenadas pela correlação.
    
    Parâmetros:
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    alvo : str, default='price'
        Nome da variável alvo.
    top_n : int, default=10
        Número de features com maior correlação para exibir no gráfico.
    metodo : str, default='pearson'
        Método estatístico: 'pearson', 'spearman' ou 'kendall'.
    cor : str, default='coolwarm'
        Paleta de cores (cmap) do heatmap. Ex: 'viridis', 'magma', 'coolwarm'.
    figsize : tuple, default=(4, 8)
        Dimensões do gráfico. Formato vertical é mais indicado.
        
    Retorna:
    -------
    pd.DataFrame
        DataFrame com TODAS as colunas numéricas ordenadas pela correlação com o alvo.
    """
    # 1. Filtra apenas colunas numéricas, ignorando IDs
    df_num = df.select_dtypes(include=[np.number]).drop(columns=['ID', 'id'], errors='ignore')
    
    if alvo not in df_num.columns:
        raise ValueError(f"A variável alvo '{alvo}' não é numérica ou não está no DataFrame.")
        
    # 2. Calcula a matriz e isola apenas a coluna do alvo
    corr = df_num.corr(method=metodo.lower())
    corr_alvo = corr[[alvo]].sort_values(by=alvo, ascending=False)
    
    # 3. Seleciona as N variáveis com maior correlação para o gráfico (incluindo o próprio alvo = 1.0)
    features_plot = corr_alvo.head(top_n + 1)
    
    # 4. Configuração e Plotagem do Gráfico
    plt.figure(figsize=figsize)
    sns.heatmap(
        features_plot,
        cmap=cor,
        annot=True,
        fmt=".3f",
        vmin=-1,
        vmax=1,
        cbar_kws={"shrink": .8},
        linewidths=.5
    )
    plt.title(f"Top {top_n} Correlações com '{alvo}'\n({metodo.capitalize()})", fontsize=13, fontweight='bold', pad=15)
    plt.ylabel('') # Remove rótulo desnecessário no eixo Y
    plt.tight_layout()
    plt.show()
    
    # 5. Retorna o DataFrame completo
    return corr_alvo


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
        plt.show()
        
    return df_nulos


analisar_nulos = analisar_valores_nulos

def plot_dispersao_tendencia(df: pd.DataFrame, x_col: str, y_col: str = 'price', figsize: tuple = (8, 5)):
    """
    Gera um gráfico de dispersão com linha de regressão (tendência linear).
    Ideal para avaliar a relação bivariada entre duas variáveis numéricas.

    Parâmetros:
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    x_col : str
        Nome da variável preditora (eixo X).
    y_col : str, default='price'
        Nome da variável alvo (eixo Y).
    figsize : tuple, default=(8, 5)
        Dimensões da figura (largura, altura).
    """
    plt.figure(figsize=figsize)
    sns.regplot(
        data=df, 
        x=x_col, 
        y=y_col,
        scatter_kws={'alpha': 0.4, 'color': 'royalblue', 's': 20},
        line_kws={'color': 'firebrick', 'linewidth': 2}
    )
    plt.title(f'Relação entre {x_col} e {y_col}', fontsize=14, fontweight='bold')
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_categoria_preco(
    df: pd.DataFrame, 
    col_cat: str, 
    col_alvo: str = 'price', 
    top_n: int = 10, 
    figsize: tuple = (12, 6)
):
    """
    Analisa a distribuição do alvo dentro das N categorias mais frequentes, 
    ordenadas de forma decrescente pela mediana.

    Parâmetros:
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    col_cat : str
        Nome da variável categórica (eixo X).
    col_alvo : str, default='price'
        Nome da variável alvo numérica (eixo Y).
    top_n : int, default=10
        Quantidade máxima de categorias (as mais frequentes) a serem exibidas.
    figsize : tuple, default=(12, 6)
        Dimensões da figura (largura, altura).
    """
    # Filtra as Top N categorias mais frequentes para não poluir o gráfico
    top_categorias = df[col_cat].value_counts().nlargest(top_n).index
    df_filtrado = df[df[col_cat].isin(top_categorias)]
    
    # Ordena as caixas (boxes) de forma decrescente pela mediana
    ordem = df_filtrado.groupby(col_cat)[col_alvo].median().sort_values(ascending=False).index

    plt.figure(figsize=figsize)
    sns.boxplot(
        data=df_filtrado, 
        x=col_cat, 
        y=col_alvo, 
        order=ordem,
        palette="Blues_r",
        showfliers=False # Oculta outliers para focar na distribuição de massa central
    )
    
    plt.title(f'Distribuição de {col_alvo} por {col_cat} (Top {top_n})', fontsize=14, fontweight='bold')
    plt.xlabel(col_cat, fontsize=12)
    plt.ylabel(col_alvo, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    sns.despine()
    plt.tight_layout()
    plt.show()


def aplicar_feature_engineering(df):
    """
    Cria features derivadas que capturam sinergia entre componentes de hardware
    e características físicas de mobilidade.

    Features geradas:
    - power_index: cpu_tier * gpu_tier * ram_gb
    - ram_per_core: ram_gb / cpu_cores
    - battery_density: battery_wh / weight_kg
    - weight_per_inch: weight_kg / display_size_in
    - cpu_turbo_range: cpu_boost_ghz - cpu_base_ghz

    Restrição: usa colunas removidas por remover_colunas_colinares
    (cpu_tier, battery_wh, weight_kg, cpu_boost_ghz), portanto precisa
    rodar ANTES da remoção.
    """
    df_copia = df.copy()
    epsilon = 1e-5

    df_copia['power_index'] = df_copia['cpu_tier'] * df_copia['gpu_tier'] * df_copia['ram_gb']
    df_copia['ram_per_core'] = df_copia['ram_gb'] / (df_copia['cpu_cores'] + epsilon)
    df_copia['battery_density'] = df_copia['battery_wh'] / (df_copia['weight_kg'] + epsilon)
    df_copia['weight_per_inch'] = df_copia['weight_kg'] / (df_copia['display_size_in'] + epsilon)
    df_copia['cpu_turbo_range'] = df_copia['cpu_boost_ghz'] - df_copia['cpu_base_ghz']

    return df_copia


def correcao_feature(df):
    """
    Cria features de interação que corrigem vieses sistemáticos:
    - Apple Tax: interações com brand_Apple (RAM, CPU, GPU, resolução)
    - Depreciação temporal: gpu/cpu power por ano de lançamento
    - Fator desktop compacto: SFF que não é laptop
    """
    df_copia = df.copy()

    # Features da cauda direita (Apple Tax)
    df_copia['apple_ram_gb'] = df_copia['brand_Apple'] * df_copia['ram_gb']
    df_copia['apple_cpu_cores'] = df_copia['brand_Apple'] * df_copia['cpu_cores']
    df_copia['apple_gpu_premium'] = df_copia['brand_Apple'] * df_copia['gpu_tier']
    df_copia['apple_resolution'] = df_copia['brand_Apple'] * df_copia['resolution_norm']

    # Features da cauda esquerda (depreciação temporal)
    df_copia['gpu_power_by_year'] = df_copia['gpu_tier'] * df_copia['release_year']
    df_copia['cpu_power_by_year'] = df_copia['cpu_cores'] * df_copia['release_year']
    df_copia['is_compact_desktop'] = df_copia['form_factor_SFF'] * (1 - df_copia['device_type_Laptop'])

    return df_copia


def pipeline_tratamento(df):
    """
    Pipeline completo de tratamento de dados para modelos baseados em árvore.
    Aplica todas as etapas de tratamento na ordem correta de dependências.

    Ordem de execução:
    1. tratar_variaveis_categoricas — encodings + resolution_norm
    2. aplicar_feature_engineering — usa colunas brutas (cpu_tier, battery_wh, etc.)
    3. correcao_feature — precisa de resolution_norm e form_factor_SFF
    4. normalizar_variaveis_numericas — escalona variáveis contínuas
    5. remover_colunas_colineares — remove redundâncias por último

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame bruto com as colunas originais do dataset.

    Retorna
    -------
    pd.DataFrame
        DataFrame tratado, pronto para modelos baseados em árvore (CatBoost, XGBoost, etc.).
    """
    df_tratado = df.copy()

    if 'price' in df_tratado.columns and 'price_log' not in df_tratado.columns:
        df_tratado['price_log'] = np.log1p(df_tratado['price'])

    df_tratado = tratar_variaveis_categoricas(df_tratado)
    df_tratado = aplicar_feature_engineering(df_tratado)
    df_tratado = correcao_feature(df_tratado)
    df_tratado = normalizar_variaveis_numericas(df_tratado)
    df_tratado = remover_colunas_colineares(df_tratado)

    print(f"=== Pipeline concluído ===")
    print(f"  {df_tratado.shape[0]} linhas × {df_tratado.shape[1]} colunas")

    return df_tratado


def reverter_escala(valores, transformacao):
    """Reverte a escala da variável alvo para US$."""
    if transformacao is None or transformacao == 'nenhum': return valores
    elif transformacao == 'log1p': return np.expm1(valores)
    elif transformacao == 'log': return np.exp(valores)
    elif transformacao == 'sqrt': return np.square(valores)
    elif transformacao == 'cbrt': return np.power(valores, 3)
    else: raise ValueError("Transformação inválida.")

def avaliar_modelo(modelo, X, y, nome_modelo, cv=5, transformacao='log1p'):
    """Aplica K-Fold CV, monitora overfitting e calcula IC 95% corrigido em US$."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Listas duplas (Treino e Validação)
    maes_val, rmses_val, r2s_val = [], [], []
    maes_train, rmses_train, r2s_train = [], [], []
    
    X_np, y_np = np.array(X), np.array(y)
    
    for train_index, val_index in kf.split(X_np):
        X_fold_train, X_fold_val = X_np[train_index], X_np[val_index]
        y_fold_train, y_fold_val = y_np[train_index], y_np[val_index]
        
        modelo.fit(X_fold_train, y_fold_train)
        
        # Previsões
        pred_fold_val = modelo.predict(X_fold_val)
        pred_fold_train = modelo.predict(X_fold_train)
        
        # Reverte a transformação para Dólares
        y_val_real = reverter_escala(y_fold_val, transformacao)
        pred_val_real = reverter_escala(pred_fold_val, transformacao)
        y_train_real = reverter_escala(y_fold_train, transformacao)
        pred_train_real = reverter_escala(pred_fold_train, transformacao)
        
        # Armazena métricas de Validação
        maes_val.append(mean_absolute_error(y_val_real, pred_val_real))
        rmses_val.append(np.sqrt(mean_squared_error(y_val_real, pred_val_real)))
        r2s_val.append(r2_score(y_val_real, pred_val_real))
        
        # Armazena métricas de Treino
        maes_train.append(mean_absolute_error(y_train_real, pred_train_real))
        rmses_train.append(np.sqrt(mean_squared_error(y_train_real, pred_train_real)))
        r2s_train.append(r2_score(y_train_real, pred_train_real))
        
    # Correção Nadeau-Bengio
    def calcular_ic(metricas):
        media = np.mean(metricas)
        variancia_amostral = np.var(metricas, ddof=1)
        prop_teste = 1 / cv
        prop_treino = 1 - prop_teste
        correcao = (1 / cv) + (prop_teste / prop_treino)
        erro_padrao_corrigido = np.sqrt(variancia_amostral * correcao)
        ic_inf, ic_sup = stats.t.interval(confidence=0.95, df=cv-1, loc=media, scale=erro_padrao_corrigido)
        return media, ic_inf, ic_sup

    # Extrai limites para Validação
    media_mae_val, mae_inf, mae_sup = calcular_ic(maes_val)
    media_rmse_val, rmse_inf, rmse_sup = calcular_ic(rmses_val)
    media_r2_val, r2_inf, r2_sup = calcular_ic(r2s_val)
    
    # Extrai médias de Treino
    media_mae_train = np.mean(maes_train)
    media_rmse_train = np.mean(rmses_train)
    media_r2_train = np.mean(r2s_train)
    
    # Margens de Erro (Validação)
    erro_mae  = abs(media_mae_val - mae_inf)
    erro_rmse = abs(media_rmse_val - rmse_inf)
    erro_r2   = abs(media_r2_val - r2_inf)

    # Gaps de Overfitting (Validação - Treino)
    gap_mae  = media_mae_val - media_mae_train
    gap_rmse = media_rmse_val - media_rmse_train
    gap_r2   = media_r2_val - media_r2_train

    # Formatação de Sinais para o Gap
    fmt_gap_mae  = f"+${gap_mae:>7.2f}" if gap_mae >= 0 else f"-${abs(gap_mae):>7.2f}"
    fmt_gap_rmse = f"+${gap_rmse:>7.2f}" if gap_rmse >= 0 else f"-${abs(gap_rmse):>7.2f}"
    fmt_gap_r2   = f"{gap_r2:>+9.4f}"

    # Exibição Estruturada Expandida
    print(f"\n{'=' * 92}")
    print(f" {nome_modelo.upper()} (K-Fold CV = {cv})")
    print(f"{'-' * 92}")
    print(f"  {'Métrica':<8} | {'Treino':<11} | {'Validação':<11} | {'Gap (V-T)':<10} | {'IC 95% (Validação)':<23} | {'Margem (±)':<10}")
    print(f"  {'-'*8} | {'-'*11} | {'-'*11} | {'-'*10} | {'-'*23} | {'-'*10}")
    print(f"  {'MAE':<8} | ${media_mae_train:>8.2f} | ${media_mae_val:>8.2f} | {fmt_gap_mae:<10} | [${mae_inf:>8.2f}, ${mae_sup:>8.2f}] | ±${erro_mae:>7.2f}")
    print(f"  {'RMSE':<8} | ${media_rmse_train:>8.2f} | ${media_rmse_val:>8.2f} | {fmt_gap_rmse:<10} | [${rmse_inf:>8.2f}, ${rmse_sup:>8.2f}] | ±${erro_rmse:>7.2f}")
    print(f"  {'R²':<8} |  {media_r2_train:>8.4f} |  {media_r2_val:>8.4f} | {fmt_gap_r2:<10} | [ {r2_inf:>8.4f},  {r2_sup:>8.4f}] |  ±{erro_r2:>7.4f}")
    print(f"{'=' * 92}\n")
    
    modelo.fit(X, y)
    return modelo

# =============================================================================
# PRÉ-PROCESSAMENTO DE DADOS
# =============================================================================

def tratar_variaveis_categoricas(df_original: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica Label Encoding em variáveis ordinais (wifi, storage_type, display_type),
    transforma resolution em pixels normalizados, remove colunas de alta cardinalidade
    e aplica One-Hot Encoding nas categóricas nominais restantes.

    Parâmetros:
    ----------
    df_original : pd.DataFrame
        DataFrame com as colunas categóricas originais.

    Retorna:
    -------
    pd.DataFrame
        DataFrame com todas as variáveis categóricas tratadas.
    """
    df = df_original.copy()

    mapa_wifi = {'Wi-Fi 5': 1, 'Wi-Fi 6': 2, 'Wi-Fi 6E': 3, 'Wi-Fi 7': 4}
    mapa_storage = {'HDD': 1, 'Hybrid': 2, 'SSD': 3, 'NVMe': 4}
    mapa_display = {'LED': 1, 'VA': 2, 'IPS': 3, 'QLED': 4, 'Mini-LED': 5, 'OLED': 6}

    df['wifi'] = df['wifi'].map(mapa_wifi)
    df['storage_type'] = df['storage_type'].map(mapa_storage)
    df['display_type'] = df['display_type'].map(mapa_display)

    if 'resolution' in df.columns:
        dimensoes = df['resolution'].str.split('x', expand=True).astype(float)
        df['total_pixels'] = dimensoes[0] * dimensoes[1]
        scaler = MinMaxScaler()
        df['resolution_norm'] = scaler.fit_transform(df[['total_pixels']])
        df = df.drop(columns=['resolution', 'total_pixels'])

    colunas_remover = ['model', 'cpu_model', 'gpu_model', 'ID']
    df = df.drop(columns=[c for c in colunas_remover if c in df.columns], errors='ignore')

    cols_categoricas = df.select_dtypes(include=['object', 'category']).columns
    df = pd.get_dummies(df, columns=cols_categoricas, drop_first=True)

    cols_booleanas = df.select_dtypes(include=['bool']).columns
    df[cols_booleanas] = df[cols_booleanas].astype(int)

    return df


def remover_colunas_colineares(
    df_original: pd.DataFrame,
    colunas_extra: list = None
) -> pd.DataFrame:
    """
    Remove colunas com alta colinearidade identificadas na EDA.

    Parâmetros:
    ----------
    df_original : pd.DataFrame
        DataFrame tratado.
    colunas_extra : list, optional
        Colunas adicionais a serem removidas.

    Retorna:
    -------
    pd.DataFrame
        DataFrame sem as colunas colineares.
    """
    colunas_colineares = [
        'price',
        'cpu_boost_ghz',
        'cpu_threads',
        'cpu_tier',
        'battery_wh',
        'psu_watts',
        'weight_kg'
    ]

    if colunas_extra:
        colunas_colineares.extend(colunas_extra)

    return df_original.drop(columns=colunas_colineares, errors='ignore')


def normalizar_variaveis_numericas(
    df_original: pd.DataFrame,
    colunas_excluidas: list = None
) -> pd.DataFrame:
    """
    Aplica MinMaxScaler nas variáveis numéricas contínuas, preservando
    a variável alvo, colunas já normalizadas e colunas binárias (One-Hot).

    Parâmetros:
    ----------
    df_original : pd.DataFrame
        DataFrame com variáveis numéricas.
    colunas_excluidas : list, optional
        Colunas adicionais a serem excluídas da normalização.

    Retorna:
    -------
    pd.DataFrame
        DataFrame com variáveis numéricas normalizadas.
    """
    df = df_original.copy()

    base_excluidas = ['price_log', 'resolution_norm']
    if colunas_excluidas:
        base_excluidas.extend(colunas_excluidas)

    colunas_binarias = [
        c for c in df.columns
        if set(df[c].dropna().unique()).issubset({0, 1})
    ]

    colunas_para_normalizar = [
        c for c in df.select_dtypes(include=['int64', 'float64', 'int32']).columns
        if c not in base_excluidas and c not in colunas_binarias
    ]

    scaler = MinMaxScaler()
    df[colunas_para_normalizar] = scaler.fit_transform(df[colunas_para_normalizar])

    return df


def preparar_dados(
    df: pd.DataFrame,
    alvo: str = 'price_log',
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """
    Pipeline completo de preparação: trata categóricas, remove colineares,
    normaliza numéricas e faz o split treino/validação.

    Parâmetros:
    ----------
    df : pd.DataFrame
        DataFrame bruto (deve conter a coluna alvo).
    alvo : str, default='price_log'
        Nome da variável alvo.
    test_size : float, default=0.2
        Proporção dos dados para validação.
    random_state : int, default=42
        Semente para reprodutibilidade.

    Retorna:
    -------
    tuple
        (X_train, X_val, y_train, y_val, feature_names)
    """
    df_proc = tratar_variaveis_categoricas(df)
    df_proc = remover_colunas_colineares(df_proc)
    df_proc = normalizar_variaveis_numericas(df_proc)

    y = df_proc[alvo]
    X = df_proc.drop(columns=[alvo])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_val, y_train, y_val, list(X.columns)


# =============================================================================
# VISUALIZAÇÃO PARA MODELAGEM
# =============================================================================

def plot_residuos(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    titulo: str = 'Análise de Resíduos',
    figsize: tuple = (14, 5)
):
    """
    Gera dois gráficos lado a lado:
    1. Resíduos vs Valores Preditos (para detectar heterocedasticidade)
    2. Q-Q Plot dos resíduos (para verificar normalidade)

    Parâmetros:
    ----------
    y_real : array-like
        Valores reais da variável alvo.
    y_pred : array-like
        Valores preditos pelo modelo.
    titulo : str, default='Análise de Resíduos'
        Título principal da figura.
    figsize : tuple, default=(14, 5)
        Dimensões da figura.
    """
    residuos = np.array(y_real) - np.array(y_pred)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.scatter(y_pred, residuos, alpha=0.3, s=10, color='steelblue', edgecolors='none')
    ax1.axhline(y=0, color='#0B2545', linestyle='--', linewidth=1.5)
    ax1.set_xlabel('Valores Preditos', fontsize=11)
    ax1.set_ylabel('Resíduos', fontsize=11)
    ax1.set_title('Resíduos vs Valores Preditos', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)

    stats.probplot(residuos, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot dos Resíduos', fontsize=12, fontweight='bold')
    ax2.get_lines()[0].set(markerfacecolor='steelblue', markeredgecolor='steelblue', markersize=3)
    ax2.get_lines()[1].set(color='#0B2545')

    fig.suptitle(titulo, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_real_vs_predito(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    titulo: str = 'Valores Reais vs Preditos',
    figsize: tuple = (7, 7)
):
    """
    Gera um scatter plot dos valores reais vs preditos com linha de referência
    (predição perfeita) para avaliação visual do modelo.

    Parâmetros:
    ----------
    y_real : array-like
        Valores reais da variável alvo.
    y_pred : array-like
        Valores preditos pelo modelo.
    titulo : str, default='Valores Reais vs Preditos'
        Título do gráfico.
    figsize : tuple, default=(7, 7)
        Dimensões da figura.
    """
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)

    plt.figure(figsize=figsize)
    plt.scatter(y_real, y_pred, alpha=0.2, s=10, color='royalblue', edgecolors='none')

    min_val = min(y_real.min(), y_pred.min())
    max_val = max(y_real.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predição Perfeita')

    plt.xlabel('Valores Reais', fontsize=12)
    plt.ylabel('Valores Preditos', fontsize=12)
    plt.title(titulo, fontsize=13, fontweight='bold')
    plt.legend(frameon=True, fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(
    modelo,
    feature_names: list,
    top_n: int = 20,
    titulo: str = None,
    figsize: tuple = (14, 10),
    cor: str = '#13579B',
    xlabel: str = None,
    annot: bool = True,
    fmt: str = '.3f',
    alpha: float = 0.85,
):
    """
    Plota a importância das features de um modelo treinado (coeficientes
    para modelos lineares ou feature_importances_ para árvores).
    Retorna o DataFrame ordenado com o Top N features.
    """
    clf = modelo
    if hasattr(modelo, 'named_steps'):
        if 'clf' in modelo.named_steps:
            clf = modelo.named_steps['clf']
        elif 'model' in modelo.named_steps:
            clf = modelo.named_steps['model']
        elif 'classifier' in modelo.named_steps:
            clf = modelo.named_steps['classifier']
        else:
            clf = modelo[-1]

    if hasattr(clf, 'feature_importances_'):
        importancias = np.asarray(clf.feature_importances_)
        tipo = 'Feature Importance'
    elif hasattr(clf, 'coef_'):
        coefs = clf.coef_
        if coefs.ndim > 1:
            coefs = coefs[0]
        importancias = np.abs(np.asarray(coefs))
        tipo = '|Coeficientes|'
    else:
        raise ValueError("O modelo não possui 'feature_importances_' nem 'coef_'.")

    feature_names_list = list(feature_names)
    if len(feature_names_list) != len(importancias):
        feature_names_list = [f"feature_{i}" for i in range(len(importancias))]

    df_imp = pd.DataFrame({
        'Feature': feature_names_list,
        'Importância': importancias
    }).sort_values('Importância', ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(
        df_imp['Feature'], df_imp['Importância'],
        color=cor, edgecolor='black', alpha=alpha,
    )

    if annot:
        max_val = df_imp['Importância'].max()
        offset = max_val * 0.02 if max_val > 0 else 0.01
        for bar, val in zip(bars, df_imp['Importância']):
            ax.text(
                bar.get_width() + offset,
                bar.get_y() + bar.get_height() / 2,
                f'{val:{fmt}}',
                va='center', fontsize=9, fontweight='bold',
            )

    ax.set_xlabel(xlabel or tipo, fontsize=12)
    ax.set_title(
        titulo or f'Top {top_n} {tipo}',
        fontsize=13, fontweight='bold',
    )
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    return df_imp.sort_values('Importância', ascending=False).reset_index(drop=True).head(top_n)

def comparar_modelos(
    modelos: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    reverter_log: bool = True,
    cv: int = 5
) -> pd.DataFrame:
    """
    Treina múltiplos modelos, avalia com métricas e cross-validation,
    e retorna uma tabela comparativa ordenada por MAE de validação.

    Parâmetros:
    ----------
    modelos : dict
        Dicionário {nome: modelo_sklearn}.
        Ex: {'XGBoost': XGBRegressor(), 'CatBoost': CatBoostRegressor()}
    X_train, y_train : array-like
        Dados de treino.
    X_val, y_val : array-like
        Dados de validação.
    reverter_log : bool, default=True
        Se True, aplica np.expm1 antes de calcular as métricas (escala US$).
    cv : int, default=5
        Número de folds para cross-validation.

    Retorna:
    -------
    pd.DataFrame
        Tabela comparativa com MAE, RMSE, R² e CV Score para cada modelo.
    """
    resultados = []

    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)

        pred_train = modelo.predict(X_train)
        pred_val = modelo.predict(X_val)

        if reverter_log:
            y_tr = np.expm1(y_train)
            y_vl = np.expm1(y_val)
            pred_train = np.expm1(pred_train)
            pred_val = np.expm1(pred_val)
        else:
            y_tr = y_train
            y_vl = y_val

        mae_val = mean_absolute_error(y_vl, pred_val)
        rmse_val = np.sqrt(mean_squared_error(y_vl, pred_val))
        r2_val = r2_score(y_vl, pred_val)

        cv_scores = cross_val_score(modelo, X_train, y_train, cv=cv, scoring='neg_mean_absolute_error')
        cv_mean = -cv_scores.mean()
        if reverter_log:
            cv_mean_display = np.expm1(cv_mean)
        else:
            cv_mean_display = cv_mean

        resultados.append({
            'Modelo': nome,
            'MAE Validação': mae_val,
            'RMSE Validação': rmse_val,
            'R² Validação': r2_val,
            f'CV MAE ({cv} folds)': cv_mean_display
        })

        print(f"  ✓ {nome} — MAE: ${mae_val:,.2f} | R²: {r2_val:.4f}")

    df_resultados = pd.DataFrame(resultados).sort_values('MAE Validação')

    fmt_cols = ['MAE Validação', 'RMSE Validação', f'CV MAE ({cv} folds)']
    for col in fmt_cols:
        df_resultados[col] = df_resultados[col].apply(lambda v: f"${v:,.2f}")
    df_resultados['R² Validação'] = df_resultados['R² Validação'].apply(lambda v: f"{v:.4f}")

    return df_resultados.reset_index(drop=True)


def pipeline_modelo_completo(
    modelo,
    X_train, y_train, X_val, y_val,
    feature_names: list = None,
    nome_modelo: str = 'Modelo',
    reverter_log: bool = True,
    plotar: bool = True
):
    """
    Pipeline completo: treina, avalia com métricas, e gera todos os
    gráficos de diagnóstico (resíduos, real vs predito, feature importance).

    Parâmetros:
    ----------
    modelo : estimator
        Modelo sklearn a ser treinado.
    X_train, y_train : array-like
        Dados de treino.
    X_val, y_val : array-like
        Dados de validação.
    feature_names : list, optional
        Nomes das features (para feature importance).
    nome_modelo : str, default='Modelo'
        Nome para exibição nos gráficos.
    reverter_log : bool, default=True
        Se True, reverte log1p para métricas em US$.
    plotar : bool, default=True
        Se True, gera os gráficos de diagnóstico.

    Retorna:
    -------
    estimator
        Modelo treinado.
    """
    modelo.fit(X_train, y_train)

    pred_train = modelo.predict(X_train)
    pred_val = modelo.predict(X_val)

    if reverter_log:
        y_tr_real = np.expm1(y_train)
        y_vl_real = np.expm1(y_val)
        pred_train_real = np.expm1(pred_train)
        pred_val_real = np.expm1(pred_val)
    else:
        y_tr_real, y_vl_real = y_train, y_val
        pred_train_real, pred_val_real = pred_train, pred_val

    mae_val = mean_absolute_error(y_vl_real, pred_val_real)
    rmse_val = np.sqrt(mean_squared_error(y_vl_real, pred_val_real))
    r2_val = r2_score(y_vl_real, pred_val_real)
    mae_train = mean_absolute_error(y_tr_real, pred_train_real)

    print(f"{'='*50}")
    print(f"  {nome_modelo}")
    print(f"{'='*50}")
    print(f"  MAE Treino:     ${mae_train:,.2f}")
    print(f"  MAE Validação:  ${mae_val:,.2f}  (Gap: ${mae_val - mae_train:,.2f})")
    print(f"  RMSE Validação: ${rmse_val:,.2f}")
    print(f"  R² Validação:   {r2_val:.4f}")
    print(f"{'='*50}\n")

    if plotar:
        plot_real_vs_predito(y_vl_real, pred_val_real, titulo=f'{nome_modelo} — Real vs Predito')
        plot_residuos(y_vl_real, pred_val_real, titulo=f'{nome_modelo} — Análise de Resíduos')
        if feature_names:
            plot_feature_importance(modelo, feature_names, titulo=f'{nome_modelo} — Feature Importance')

    return modelo


def plotar_importancia_shap(dict_shap, features, top_n=15, 
                            cor_destaque="#0B2545", cor_base="#3E7CC4", 
                            fundo_figura='#F8F9FA', cor_texto='#2C3E50'):
    """
    Plota o ranking de importância das variáveis usando valores SHAP.
    
    Parâmetros:
    - dict_shap: Dicionário contendo { 'Nome do Modelo': matriz_de_valores_shap }
    - features: Lista ou Index do Pandas com os nomes das colunas
    - top_n: Quantidade de variáveis para exibir no ranking
    - cor_destaque: Cor das top 3 barras (padrão: Azul forte)
    - cor_base: Cor das demais barras (padrão: Azul claro)
    - fundo_figura: Cor do fundo do gráfico
    - cor_texto: Cor dos títulos e rótulos
    """
    n_modelos = len(dict_shap)
    cols = 3 if n_modelos >= 3 else n_modelos
    rows = math.ceil(n_modelos / cols)
    
    fig_width = 8 * cols
    fig_height = 7 * rows
    
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), facecolor=fundo_figura)
    
    y_title_pos = 0.98 if rows == 1 else 0.96
    fig.suptitle(f'Análise SHAP: Top {top_n} Variáveis por Impacto Médio (|SHAP|)',
                 fontsize=22, fontweight='bold', y=y_title_pos, color=cor_texto)
    
    if n_modelos == 1:
        axes = [axes]
    elif rows > 1 or cols > 1:
        axes = axes.flatten()

    for i, (nome, shap_vals) in enumerate(dict_shap.items()):
        ax = axes[i]
        ax.set_facecolor(fundo_figura)
        
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)

        ranking_idx = np.argsort(mean_abs_shap)[::-1][:top_n]
        top_features = [features[j] for j in ranking_idx]
        top_values = mean_abs_shap[ranking_idx]

        # Aplica as cores personalizáveis passadas na função
        cores = [cor_destaque if idx < 3 else cor_base for idx in range(len(top_features))][::-1]

        y_pos = range(len(top_features)-1, -1, -1)
        bars = ax.barh(y_pos, top_values, color=cores, edgecolor='white', height=0.7)
        
        ax.bar_label(bars, fmt='%.4f', padding=6, color=cor_texto, fontsize=10, fontweight='bold')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features, fontsize=11, fontweight='500', color='#333333')
        ax.set_xlim(0, max(top_values) * 1.15)
        
        ax.set_xlabel('Impacto Médio |SHAP|', fontsize=12, fontweight='bold', color='#555555', labelpad=10)
        ax.set_title(f' {nome}', fontsize=16, fontweight='bold', pad=15, color=cor_texto)
        
        ax.grid(True, axis='x', linestyle='--', alpha=0.6, color='#BDC3C7')
        ax.grid(False, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#DDDDDD')
        ax.spines['bottom'].set_color('#DDDDDD')

    if n_modelos > 1:
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.88 if rows > 1 else 0.85, hspace=0.35, wspace=0.25) 
    plt.show()


def plotar_top_coeficientes(df_coefs, top_n=10, titulo='Top Coeficientes do Modelo'):
    df_c = df_coefs.copy()
    
    # 1. Isolando o Top N pelo valor absoluto
    df_c['|Coeficiente|'] = df_c['Coeficiente'].abs()
    top_features = df_c.nlargest(top_n, '|Coeficiente|').sort_values('|Coeficiente|', ascending=True)

    # 2. Plotagem do Gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_top = ['#0B2545' if c >= 0 else '#48CAE4' for c in top_features['Coeficiente']]

    bars = ax.barh(top_features['Feature'], top_features['Coeficiente'], 
                   color=colors_top, edgecolor='white', height=0.7, alpha=0.85)

    # Adicionando os textos
    for bar, val in zip(bars, top_features['Coeficiente']):
        offset = 0.01 if val >= 0 else -0.01
        alinhamento = 'left' if val >= 0 else 'right'
        
        ax.text(
            val + offset, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', ha=alinhamento, fontsize=9, fontweight='bold', color='#333'
        )

    # --- A CORREÇÃO DE MARGEM ENTRA AQUI ---
    min_val = top_features['Coeficiente'].min()
    max_val = top_features['Coeficiente'].max()
    margem = (max_val - min_val) * 0.15 # Adiciona 15% de respiro
    
    # Garante que o limite esquerdo acomode o texto da barra negativa sem esmagar o eixo Y
    limite_esq = min(0, min_val) - margem
    limite_dir = max(0, max_val) + margem
    ax.set_xlim(limite_esq, limite_dir)
    # ---------------------------------------

    ax.axvline(x=0, color='black', linewidth=1.2, linestyle='-')
    ax.set_xlabel('Coeficiente', fontsize=12, fontweight='bold')
    ax.set_title(titulo, fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

    # 4. Display dos dados formatados
    print(f"\n--- Tabela das Top {top_n} Features ---")
    df_display = top_features.sort_values('|Coeficiente|', ascending=False)[['Feature', 'Coeficiente']]
    df_display.reset_index(drop=True, inplace=True)
    display(df_display)

def plotar_real_vs_predito(modelos_dict, X, y_true_log, cores_dict=None, titulo_geral='Comparação dos Modelos — Valores Reais vs Preditos'):
    """
    Gera scatter plots comparando os preços reais vs preditos para múltiplos modelos.
    
    Parâmetros:
    - modelos_dict: Dicionário com os nomes dos modelos e os estimadores treinados.
    - X: Variáveis preditoras (ex: X_tree).
    - y_true_log: Target verdadeiro em logaritmo (ex: y_tree).
    - cores_dict: Dicionário mapeando o nome do modelo para a sua cor (ex: {'XGBoost': '#0B2545'}).
    - titulo_geral: Título principal da figura.
    """
    y_real_usd = np.expm1(y_true_log)
    n_modelos = len(modelos_dict)
    
    fig, axes = plt.subplots(1, n_modelos, figsize=(7 * n_modelos, 6))
    if n_modelos == 1:
        axes = [axes]
        
    # Garante que o dicionário de cores exista mesmo se o usuário não passar
    if cores_dict is None:
        cores_dict = {}
        
    cores_fallback = ['#0B2545', '#13579B', '#0096C7', '#3E7CC4', '#48CAE4']

    for idx, (nome, modelo) in enumerate(modelos_dict.items()):
        ax = axes[idx]
        
        pred_log = modelo.predict(X)
        y_pred = np.expm1(pred_log)
        
        # Puxa a cor do seu dicionário, ou usa uma reserva caso o nome não exista lá
        cor = cores_dict.get(nome, cores_fallback[idx % len(cores_fallback)])
        
        ax.scatter(y_real_usd, y_pred, alpha=0.15, s=8, color=cor, edgecolors='none')
        
        min_val = min(y_real_usd.min(), y_pred.min())
        max_val = max(y_real_usd.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='Predição Perfeita')
        
        mae = mean_absolute_error(y_real_usd, y_pred)
        r2 = r2_score(y_real_usd, y_pred)
        
        ax.set_xlabel('Preço Real (US$)', fontsize=11)
        ax.set_ylabel('Preço Predito (US$)', fontsize=11)
        ax.set_title(nome, fontsize=12, fontweight='bold')
        ax.text(0.05, 0.95, f'MAE: ${mae:,.2f}\nR²: {r2:.4f}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, linestyle='--', alpha=0.4)

    fig.suptitle(titulo_geral, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def plotar_distribuicao_residuos(modelos_dict, X, y_true_log, cores_dict=None, titulo_geral='Distribuição dos Resíduos por Modelo'):
    """
    Gera histogramas da distribuição dos resíduos para múltiplos modelos,
    além de imprimir as estatísticas de dispersão e forma (Desvio Padrão, Assimetria, Curtose).
    
    Parâmetros:
    - modelos_dict: Dicionário com os nomes dos modelos e os estimadores treinados.
    - X: Variáveis preditoras.
    - y_true_log: Target verdadeiro em logaritmo.
    - cores_dict: Dicionário mapeando o nome do modelo para a sua cor.
    - titulo_geral: Título principal da figura.
    """
    y_real_usd = np.expm1(y_true_log)
    n_modelos = len(modelos_dict)
    
    fig, axes = plt.subplots(1, n_modelos, figsize=(7 * n_modelos, 5))
    if n_modelos == 1:
        axes = [axes]
        
    if cores_dict is None:
        cores_dict = {}
        
    cores_fallback = ['#0B2545', '#13579B', '#0096C7', '#3E7CC4', '#48CAE4']
    
    print(f"=== ESTATÍSTICAS DOS RESÍDUOS ===")

    for idx, (nome, modelo) in enumerate(modelos_dict.items()):
        ax = axes[idx]
        
        # Previsão e cálculo do resíduo (Real - Predito)
        pred_log = modelo.predict(X)
        y_pred = np.expm1(pred_log)
        residuos = y_real_usd - y_pred
        
        # Converte para Series do Pandas para usar métodos estatísticos facilitados
        residuos_s = pd.Series(residuos)
        media_res = residuos_s.mean()
        std_res = residuos_s.std()
        skew_res = residuos_s.skew()
        kurt_res = residuos_s.kurtosis()
        
        # Print das estatísticas solicitadas
        print(f"\n[{nome}]")
        print(f"  Média         : ${media_res:.2f}")
        print(f"  Desvio Padrão : ${std_res:.2f}")
        print(f"  Assimetria    : {skew_res:.3f}")
        print(f"  Curtose       : {kurt_res:.3f}")
        
        cor = cores_dict.get(nome, cores_fallback[idx % len(cores_fallback)])
        
        # Histograma com curva de densidade (KDE)
        sns.histplot(residuos_s, kde=True, bins=50, color=cor, ax=ax, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
        
        # Formatação do eixo e títulos
        ax.set_xlabel('Resíduo (US$)', fontsize=11)
        ax.set_ylabel('Frequência', fontsize=11)
        ax.set_title(nome, fontsize=12, fontweight='bold')
        
        # Caixinha de texto interna no gráfico (agora com Curtose incluída)
        ax.text(0.95, 0.95, f'Média: ${media_res:,.2f}\nStd: ${std_res:,.2f}\nSkew: {skew_res:.3f}\nKurt: {kurt_res:.3f}',
                transform=ax.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    print("\n" + "="*33 + "\n")
    
    fig.suptitle(titulo_geral, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def plotar_qq_plot_residuos(modelos_dict, X, y_true_log, cores_dict=None, titulo_geral='QQ Plot dos Resíduos por Modelo'):
    """
    Gera gráficos QQ (Quantile-Quantile) dos resíduos para verificar a normalidade dos erros.
    
    Parâmetros:
    - modelos_dict: Dicionário com os nomes dos modelos e os estimadores treinados.
    - X: Variáveis preditoras.
    - y_true_log: Target verdadeiro em logaritmo.
    - cores_dict: Dicionário mapeando o nome do modelo para a sua cor.
    - titulo_geral: Título principal da figura.
    """
    y_real_usd = np.expm1(y_true_log)
    n_modelos = len(modelos_dict)
    
    fig, axes = plt.subplots(1, n_modelos, figsize=(7 * n_modelos, 5))
    if n_modelos == 1:
        axes = [axes]
        
    if cores_dict is None:
        cores_dict = {}
        
    cores_fallback = ['#0B2545', '#13579B', '#0096C7', '#3E7CC4', '#48CAE4']

    for idx, (nome, modelo) in enumerate(modelos_dict.items()):
        ax = axes[idx]
        
        # Previsão e cálculo do resíduo
        pred_log = modelo.predict(X)
        y_pred = np.expm1(pred_log)
        residuos = y_real_usd - y_pred
        
        cor = cores_dict.get(nome, cores_fallback[idx % len(cores_fallback)])
        
        # Gera o QQ plot no eixo específico
        stats.probplot(residuos, dist="norm", plot=ax)
        
        # O scipy cria 2 linhas no gráfico: [0] são os pontos do resíduo, [1] é a linha de tendência ideal
        pontos = ax.get_lines()[0]
        linha_ideal = ax.get_lines()[1]
        
        # Customização visual
        pontos.set(markerfacecolor=cor, markeredgecolor=cor, markersize=3, alpha=0.5)
        linha_ideal.set(color='firebrick', linewidth=2)
        
        # Sobrescrevendo os textos gerados automaticamente pelo scipy
        ax.set_title(f'QQ Plot — {nome}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Quantis Teóricos', fontsize=11)
        ax.set_ylabel('Quantis Amostrais', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.4)

    fig.suptitle(titulo_geral, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def plotar_erros_por_faixa(modelos_dict, X, y_true_log, cores_dict=None, quantis=4):
    """
    Segmenta os preços reais em quantis e avalia RMSE e MAPE por faixa de preço.
    Gera tabela estilizada e gráficos de barras comparativos.
    """
    y_real_usd = np.expm1(y_true_log)
    
    # 1. Segmentar os preços em Faixas (Quantis)
    df_analise = pd.DataFrame({'Preço Real': y_real_usd})
    df_analise['Faixa de Preço'] = pd.qcut(df_analise['Preço Real'], q=quantis, 
                                           labels=['Q1 (Mais Baratos)', 'Q2 (Intermediários)', 
                                                   'Q3 (Avançados)', 'Q4 (Premium)'])
    
    # 2. Calcular previsões e agregar os erros
    resultados = []
    
    for nome, modelo in modelos_dict.items():
        y_pred = np.expm1(modelo.predict(X))
        df_analise[f'Pred_{nome}'] = y_pred
        
        for faixa in df_analise['Faixa de Preço'].unique():
            mask = df_analise['Faixa de Preço'] == faixa
            y_real_faixa = df_analise.loc[mask, 'Preço Real']
            y_pred_faixa = df_analise.loc[mask, f'Pred_{nome}']
            
            rmse = np.sqrt(mean_squared_error(y_real_faixa, y_pred_faixa))
            mape = mean_absolute_percentage_error(y_real_faixa, y_pred_faixa) * 100
            
            resultados.append({
                'Modelo': nome,
                'Faixa de Preço': faixa,
                'RMSE ($)': rmse,
                'MAPE (%)': mape,
                'Vol. de Dados': mask.sum()
            })
            
    df_resultados = pd.DataFrame(resultados).sort_values('Faixa de Preço')
    
    # 3. Exibir a Tabela de Calor
    print("=== Tabela de Erros por Faixa de Preço ===")
    tabela_pivot_mape = df_resultados.pivot(index='Faixa de Preço', columns='Modelo', values='MAPE (%)')
    tabela_pivot_rmse = df_resultados.pivot(index='Faixa de Preço', columns='Modelo', values='RMSE ($)')
    
    display(tabela_pivot_mape.style.background_gradient(cmap='OrRd', axis=1)\
            .format("{:.2f}%").set_caption("MAPE (%) - Menor é Melhor"))
    
    display(tabela_pivot_rmse.style.background_gradient(cmap='OrRd', axis=1)\
            .format("${:.2f}").set_caption("RMSE ($) - Menor é Melhor"))
    
    # 4. Plotagem dos Gráficos
    if cores_dict is None:
        cores_dict = {'XGBoost Otimizado': '#0B2545', 'LightGBM Otimizado': '#0096C7', 'CatBoost Otimizado': '#5DADE2'}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gráfico 1: MAPE
    sns.barplot(data=df_resultados, x='Faixa de Preço', y='MAPE (%)', hue='Modelo', 
                palette=cores_dict, ax=axes[0], edgecolor='black', alpha=0.8)
    axes[0].set_title('Erro Percentual Médio (MAPE) por Faixa', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('MAPE (%)', fontsize=11)
    axes[0].grid(axis='y', linestyle='--', alpha=0.4)
    axes[0].legend(loc='upper right', fontsize=9)
    
    # Gráfico 2: RMSE
    sns.barplot(data=df_resultados, x='Faixa de Preço', y='RMSE ($)', hue='Modelo', 
                palette=cores_dict, ax=axes[1], edgecolor='black', alpha=0.8)
    axes[1].set_title('Erro Absoluto (RMSE) por Faixa', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('RMSE em Dólares ($)', fontsize=11)
    axes[1].grid(axis='y', linestyle='--', alpha=0.4)
    axes[1].legend(loc='upper left', fontsize=9)
    
    fig.suptitle('Desempenho dos Modelos Segmentado por Custo do Hardware', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plotar_boxplot_faixa(df, col_x='ram_gb', col_y='price', titulo=None, cor='#2E86C1'):
    """
    Plota a distribuição e a variação exata de preços para cada categoria de RAM.
    Permite customizar o título e a cor principal do gráfico.
    """
    plt.figure(figsize=(12, 6))

    # Boxplot base com a cor dinâmica
    sns.boxplot(data=df, x=col_x, y=col_y, color=cor,
                flierprops={'marker': 'o', 'markerfacecolor': 'red', 'alpha': 0.5})

    # Lógica do título: usa o fornecido ou monta um padrão
    titulo_final = titulo if titulo else f'Variação de Preço por Quantidade de {col_x}'

    plt.title(titulo_final, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel(col_x, fontsize=12, fontweight='bold')
    plt.ylabel(f'{col_y} (US$)', fontsize=12, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_matriz_correlacao(df, titulo='Matriz de Correlação', triangulo=True,
                           cmap='coolwarm', figsize=(18, 14), annot_size=9):
    """
    Plota heatmap da matriz de correlação de variáveis numéricas.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com colunas numéricas.
    titulo : str, default='Matriz de Correlação'
        Título do gráfico.
    triangulo : bool or str, default=True
        - True ou 'inferior': exibe apenas o triângulo inferior.
        - 'superior': exibe apenas o triângulo superior.
        - False ou 'completa': exibe a matriz completa.
    cmap : str, default='coolwarm'
        Paleta de cores do heatmap. Ex: 'RdBu_r', 'viridis', 'magma'.
    figsize : tuple, default=(18, 14)
        Dimensões da figura.
    annot_size : int, default=9
        Tamanho da fonte das anotações.

    Retorna
    -------
    pd.DataFrame
        Matriz de correlação calculada.
    """
    df_num = df.select_dtypes(include=[np.number])
    corr = df_num.corr()

    mascara = None
    if triangulo in (True, 'inferior'):
        mascara = np.triu(np.ones_like(corr, dtype=bool), k=1)
    elif triangulo == 'superior':
        mascara = np.tril(np.ones_like(corr, dtype=bool), k=-1)

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr, annot=True, fmt='.2f', cmap=cmap,
        vmin=-1, vmax=1, center=0, square=True,
        linewidths=.5, cbar_kws={'shrink': .75},
        mask=mascara, annot_kws={'size': annot_size}
    )
    plt.title(titulo, fontsize=16, pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    return corr


def plot_nulos_historico(df, limite=40, cor='#13579B', cor_limite='#0096C7',
                         figsize=(10, 7)):
    """
    Plota percentual de valores ausentes por variável em barras horizontais.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado.
    limite : int or float, default=40
        Valor da linha de corte vertical (%).
    cor : str, default='#13579B'
        Cor das barras.
    cor_limite : str, default='red'
        Cor da linha de corte.
    figsize : tuple, default=(10, 7)
        Dimensões da figura.

    Retorna
    -------
    pd.Series
        Percentual de nulos por coluna (apenas colunas com nulos), ordenado.
    """
    nulos_pct = (df.isnull().sum() / len(df) * 100)
    nulos_pct = nulos_pct[nulos_pct > 0].sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(nulos_pct.index, nulos_pct.values,
                   color=cor, edgecolor='black', alpha=0.85)

    for bar, val in zip(bars, nulos_pct.values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', fontsize=9)

    ax.axvline(x=limite, color=cor_limite, linestyle='--', alpha=0.7,
               label=f'Corte {limite}%')
    ax.set_xlabel('% de Nulos', fontsize=11)
    ax.set_title('Percentual de Valores Ausentes', fontsize=14, fontweight='bold')
    ax.legend()
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()

    return nulos_pct


def plot_analise_temporal(df, col_target='RainTomorrow', col_mes='Month',
                          titulo=None,
                          cor_acima='#0B2545', cor_abaixo='#5DADE2',
                          cor_media='#0096C7', cor_amostras='#5A6878',
                          figsize=(12, 6)):
    """
    Plota a taxa do alvo por mês com barras coloridas, linha de média global
    e eixo secundário com volume de amostras.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com colunas de mês e target.
    col_target : str, default='RainTomorrow'
        Coluna alvo (binária ou numérica).
    col_mes : str, default='Month'
        Coluna com o número do mês (1–12).
    titulo : str, optional
        Título principal do gráfico. Se None, gera automaticamente.
    cor_acima : str, default='#0B2545'
        Cor das barras acima da média global.
    cor_abaixo : str, default='#3498DB'
        Cor das barras abaixo da média global.
    cor_media : str, default='#2C3E50'
        Cor da linha tracejada de média global.
    cor_amostras : str, default='#7F8C8D'
        Cor da linha de volume de amostras (eixo secundário).
    figsize : tuple, default=(12, 6)
        Dimensões da figura.

    Retorna
    -------
    pd.DataFrame
        DataFrame com 'mean' (%) e 'count' por mês.
    """
    taxa_mes = df.groupby(col_mes)[col_target].agg(['mean', 'count'])
    taxa_mes['mean'] *= 100
    media_global = df[col_target].mean() * 100

    titulo_final = titulo or f'Sazonalidade — Taxa de {col_target} por Mês'

    fig, ax1 = plt.subplots(figsize=figsize)

    cores = [cor_acima if v > media_global else cor_abaixo
             for v in taxa_mes['mean']]
    bars = ax1.bar(taxa_mes.index, taxa_mes['mean'],
                   color=cores, edgecolor='black', alpha=0.85, width=0.6, zorder=3)

    for bar, val in zip(bars, taxa_mes['mean']):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.axhline(media_global, color=cor_media, linestyle='--', linewidth=1.5,
                label=f'Média Global ({media_global:.1f}%)', zorder=4)

    ax2 = ax1.twinx()
    ax2.plot(taxa_mes.index, taxa_mes['count'],
             color=cor_amostras, marker='o', linewidth=2, markersize=6,
             label='Amostras', zorder=5)
    ax2.set_ylabel('Nº de Amostras', fontsize=11, color=cor_amostras)
    ax2.tick_params(axis='y', labelcolor=cor_amostras)

    meses_nomes = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
                   7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'}
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels([meses_nomes.get(m, str(m)) for m in range(1, 13)], fontsize=10)
    ax1.set_ylabel('Taxa (%)', fontsize=11)
    ax1.set_xlabel('Mês', fontsize=11)
    ax1.set_title(titulo_final, fontsize=14, fontweight='bold', pad=15)
    ax1.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax1.set_axisbelow(True)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', frameon=True)

    plt.tight_layout()
    plt.show()

    return taxa_mes


def plot_grid_distribuicao(
    df: pd.DataFrame,
    features: list,
    titulo: str = 'Distribuição das Features',
    cor: str = '#13579B',
    bins: str = 'sturges',
    figsize: tuple = (16, 12),
):
    """
    Plota um grid 2×2 com boxplot (topo) + histograma com KDE (base)
    para cada feature numérica, destacando média, mediana e moda.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    features : list
        Lista com até 4 colunas numéricas a serem plotadas.
    titulo : str, default='Distribuição das Features'
        Título principal da figura.
    cor : str, default='steelblue'
        Cor dos boxplots e histogramas.
    bins : str ou int, default='sturges'
        Método ou número de bins do histograma.
    figsize : tuple, default=(16, 12)
        Dimensões da figura.
    """
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=figsize)
    outer_gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.2)

    for i, col in enumerate(features[:4]):
        row_idx = i // 2
        col_idx = i % 2

        inner_gs = outer_gs[row_idx, col_idx].subgridspec(
            2, 1, height_ratios=[1, 4], hspace=0.05
        )

        ax_box = fig.add_subplot(inner_gs[0])
        ax_hist = fig.add_subplot(inner_gs[1], sharex=ax_box)

        dados = df[col].dropna()
        media = dados.mean()
        mediana = dados.median()
        modas = dados.mode()
        moda = modas[0] if not modas.empty else None

        sns.boxplot(x=dados, ax=ax_box, color=cor, fliersize=4)
        ax_box.axvline(media, color='#0B2545', linestyle='--', linewidth=1.5)
        ax_box.axvline(mediana, color='#0096C7', linestyle='--', linewidth=1.5)
        if moda is not None:
            ax_box.axvline(moda, color='#3E7CC4', linestyle='--', linewidth=1.5)

        ax_box.set_title(f'Distribuição: {col}', fontsize=13, fontweight='bold', pad=10)
        ax_box.set_xlabel('')
        ax_box.grid(True, linestyle='--', alpha=0.7)
        plt.setp(ax_box.get_xticklabels(), visible=False)

        sns.histplot(
            x=dados, bins=bins, kde=True, ax=ax_hist,
            color=cor, edgecolor='black', alpha=0.6,
        )

        ax_hist.axvline(media, color='#0B2545', linestyle='--', linewidth=2, label=f'Média: {media:.1f}')
        ax_hist.axvline(mediana, color='#0096C7', linestyle='--', linewidth=2, label=f'Mediana: {mediana:.1f}')
        if moda is not None:
            ax_hist.axvline(moda, color='#3E7CC4', linestyle='--', linewidth=2, label=f'Moda: {moda:.1f}')

        ax_hist.legend(frameon=True, fontsize=9, loc='upper right')
        ax_hist.set_xlabel(col, fontsize=11)
        ax_hist.set_ylabel('Frequência', fontsize=11)
        ax_hist.grid(True, linestyle='--', alpha=0.7)

    plt.suptitle(titulo, fontsize=18, fontweight='bold', y=0.96)
    plt.show()


def plot_grid_boxplot_alvo(
    df: pd.DataFrame,
    features: list,
    alvo: str = 'RainTomorrow',
    hue: str = 'RainToday',
    titulo: str = None,
    paleta: str = 'Blues',
    labels_x: list = None,
    figsize: tuple = (16, 12),
):
    """
    Plota um grid 2×2 de boxplots agrupados pelo alvo e coloridos por hue,
    útil para visualizar a separabilidade das classes.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    features : list
        Lista com até 4 colunas numéricas.
    alvo : str, default='RainTomorrow'
        Coluna do eixo X (variável alvo).
    hue : str, default='RainToday'
        Coluna de agrupamento por cor.
    titulo : str, optional
        Título principal. Se None, gera automaticamente.
    paleta : str, default='Set2'
        Paleta de cores do seaborn.
    labels_x : list, optional
        Rótulos personalizados para o eixo X. Se None, usa os valores brutos.
    figsize : tuple, default=(16, 12)
        Dimensões da figura.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    titulo_final = titulo or f'Top 4 Features — Distribuição por {alvo}'

    for i, var in enumerate(features[:4]):
        sns.boxplot(
            x=alvo, y=var, hue=hue,
            data=df, ax=axes[i], palette=paleta,
            showmeans=True,
            meanprops={
                'marker': 'D', 'markerfacecolor': 'white',
                'markeredgecolor': 'black', 'markersize': 6,
            },
        )
        axes[i].set_title(f'{var} vs {alvo}', fontsize=13, fontweight='bold', pad=10)
        axes[i].set_xlabel('')
        axes[i].set_ylabel(var, fontsize=11)
        axes[i].grid(axis='y', linestyle='--', alpha=0.5)

        if labels_x is not None:
            axes[i].set_xticklabels(labels_x)

        if i == 0:
            axes[i].legend(title=hue, loc='upper left')
        else:
            axes[i].get_legend().remove()

    plt.suptitle(titulo_final, fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()


def teste_significancia_chi2(
    df: pd.DataFrame,
    colunas_nominais: list,
    alvo: str = 'RainTomorrow',
) -> pd.DataFrame:
    """
    Aplica o teste Qui-Quadrado e calcula o V de Cramér para medir a
    força de associação entre variáveis nominais e a variável alvo.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    colunas_nominais : list
        Lista de colunas categóricas nominais a serem testadas.
    alvo : str, default='RainTomorrow'
        Nome da variável alvo (binária ou categórica).

    Retorna
    -------
    pd.DataFrame
        Tabela com Feature, Categorias, Chi², p-valor e V de Cramér,
        ordenada de forma decrescente pelo V de Cramér.
    """
    resultados = []

    for col in colunas_nominais:
        sub = df[[col, alvo]].dropna()
        tab = pd.crosstab(sub[col], sub[alvo])

        chi2, p_valor, dof, _ = stats.chi2_contingency(tab)
        n = tab.sum().sum()
        k = min(tab.shape)
        cramers_v = np.sqrt(chi2 / (n * (k - 1)))

        resultados.append({
            'Feature': col,
            'Categorias': sub[col].nunique(),
            'Chi²': round(chi2, 2),
            'p-valor': f'{p_valor:.4e}',
            'V de Cramér': round(cramers_v, 4),
        })

    return (
        pd.DataFrame(resultados)
        .sort_values('V de Cramér', ascending=False)
        .reset_index(drop=True)
    )

def avaliar_classificador(modelo, X_train, y_train, X_test, y_test, nome_modelo: str, cv: int = 3, thresholds_dict: dict = None, groups=None, avaliadores: str = 'todos'):
    """
    Avalia um modelo de classificação binária via validação cruzada no Treino
    (GroupKFold se `groups` fornecido, senão StratifiedKFold) e teste cego em `X_test`.
    Exibe a média real dos folds de treino do CV (`Treino CV`), média dos folds de validação
    (`Val CV` + IC 95%), desempenho em `X_test` (`Teste`) e o gap (`Teste - Treino CV`).

    avaliadores:
      - 'independentes': apenas métricas de ranqueamento (ROC-AUC, GINI, PR-AUC, KS);
      - 'dependentes'  : apenas métricas do corte (Accuracy, Precision, Recall);
      - 'todos'        : ambos os blocos.
    """
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, accuracy_score,
        precision_score, recall_score
    )
    from scipy.stats import ks_2samp
    from sklearn.model_selection import StratifiedKFold, GroupKFold

    opcoes_validas = ('independentes', 'dependentes', 'todos')
    if avaliadores not in opcoes_validas:
        raise ValueError(f"`avaliadores` deve ser um de {opcoes_validas}. Recebido: '{avaliadores}'")
    usar_indep = avaliadores in ('independentes', 'todos')
    usar_dep   = avaliadores in ('dependentes', 'todos')

    threshold = 0.5
    if thresholds_dict is not None and nome_modelo in thresholds_dict:
        threshold = float(thresholds_dict[nome_modelo])

    y_tr_arr = np.asarray(y_train)
    y_te_arr = np.asarray(y_test)

    if groups is not None:
        cv_splitter = GroupKFold(n_splits=cv) if isinstance(cv, int) else cv
        splits = cv_splitter.split(X_train, y_tr_arr, groups=groups)
        cv_desc = f"GroupKFold CV = {getattr(cv_splitter, 'n_splits', cv)}"
    else:
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42) if isinstance(cv, int) else cv
        splits = cv_splitter.split(X_train, y_tr_arr)
        cv_desc = f"CV = {getattr(cv_splitter, 'n_splits', cv)}"

    print(f"{'='*108}")
    desc_th = f" | Threshold = {threshold:.4f}" if usar_dep else ""
    print(f" {nome_modelo.upper()} ({cv_desc}{desc_th})")
    print(f"{'-'*108}")

    prop_tr = float(np.mean(y_tr_arr)) * 100
    prop_te = float(np.mean(y_te_arr)) * 100
    print(f" Taxa Base de Fraude (Target=1): Treino = {prop_tr:.2f}% | Teste = {prop_te:.2f}%")
    print(f"{'-'*108}")

    cv_tr_roc, cv_tr_gini, cv_tr_pr, cv_tr_ks = [], [], [], []
    cv_tr_acc, cv_tr_prec, cv_tr_rec = [], [], []
    cv_va_roc, cv_va_gini, cv_va_pr, cv_va_ks = [], [], [], []
    cv_va_acc, cv_va_prec, cv_va_rec = [], [], []

    for train_idx, val_idx in splits:
        if hasattr(X_train, 'iloc'):
            X_f_tr, X_f_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
        else:
            X_f_tr, X_f_va = X_train[train_idx], X_train[val_idx]

        y_f_tr, y_f_va = y_tr_arr[train_idx], y_tr_arr[val_idx]

        modelo.fit(X_f_tr, y_f_tr)

        # Métricas nos folds de treino do CV (Treino CV)
        probs_f_tr = modelo.predict_proba(X_f_tr)[:, 1]
        if usar_indep:
            roc_ftr = roc_auc_score(y_f_tr, probs_f_tr)
            cv_tr_roc.append(roc_ftr)
            cv_tr_gini.append(2 * roc_ftr - 1)
            cv_tr_pr.append(average_precision_score(y_f_tr, probs_f_tr))
            pos_ftr, neg_ftr = probs_f_tr[y_f_tr == 1], probs_f_tr[y_f_tr == 0]
            cv_tr_ks.append(ks_2samp(pos_ftr, neg_ftr).statistic if (len(pos_ftr) > 0 and len(neg_ftr) > 0) else 0.0)
        if usar_dep:
            preds_f_tr = (probs_f_tr >= threshold).astype(int)
            cv_tr_acc.append(accuracy_score(y_f_tr, preds_f_tr))
            cv_tr_prec.append(precision_score(y_f_tr, preds_f_tr, zero_division=0))
            cv_tr_rec.append(recall_score(y_f_tr, preds_f_tr, zero_division=0))

        # Métricas nos folds de validação do CV (Val CV)
        probs_f_va = modelo.predict_proba(X_f_va)[:, 1]
        if usar_indep:
            roc_fva = roc_auc_score(y_f_va, probs_f_va)
            cv_va_roc.append(roc_fva)
            cv_va_gini.append(2 * roc_fva - 1)
            cv_va_pr.append(average_precision_score(y_f_va, probs_f_va))
            pos_fva, neg_fva = probs_f_va[y_f_va == 1], probs_f_va[y_f_va == 0]
            cv_va_ks.append(ks_2samp(pos_fva, neg_fva).statistic if (len(pos_fva) > 0 and len(neg_fva) > 0) else 0.0)
        if usar_dep:
            preds_f_va = (probs_f_va >= threshold).astype(int)
            cv_va_acc.append(accuracy_score(y_f_va, preds_f_va))
            cv_va_prec.append(precision_score(y_f_va, preds_f_va, zero_division=0))
            cv_va_rec.append(recall_score(y_f_va, preds_f_va, zero_division=0))

    # Treino final em 100% de X_train e avaliação no X_test de validação (antes do OOT)
    modelo.fit(X_train, y_tr_arr)
    prob_test = modelo.predict_proba(X_test)[:, 1]

    def format_row(name, tr_cv, va_cv, val_20, ic="-"):
        gap = val_20 - tr_cv
        gap_str = f"{'+' if gap > 0 else ''}{gap:.4f}"
        return f" {name:<10} | {tr_cv:.4f}     | {va_cv:.4f}     | {val_20:.4f}     | {gap_str:<11} | {ic}"

    header = f" {'Metrica':<10} | {'Treino':<10} | {'Teste':<10} | {'Validacao':<10} | {'Gap(Va-Tr)':<11} | {'IC 95% (Teste CV)'}"

    if usar_indep:
        ic_roc = f"[{np.mean(cv_va_roc) - 2*np.std(cv_va_roc):.4f}, {np.mean(cv_va_roc) + 2*np.std(cv_va_roc):.4f}]"
        ic_gini = f"[{np.mean(cv_va_gini) - 2*np.std(cv_va_gini):.4f}, {np.mean(cv_va_gini) + 2*np.std(cv_va_gini):.4f}]"

        roc_te = roc_auc_score(y_te_arr, prob_test)
        gini_te = 2 * roc_te - 1
        pr_te = average_precision_score(y_te_arr, prob_test)
        pos_te, neg_te = prob_test[y_te_arr == 1], prob_test[y_te_arr == 0]
        ks_te = ks_2samp(pos_te, neg_te).statistic if (len(pos_te) > 0 and len(neg_te) > 0) else 0.0

        print("\n [ METRICAS INDEPENDENTES DO THRESHOLD (Probabilidades) ]")
        print(header)
        print("-" * len(header))
        print(format_row("ROC-AUC", np.mean(cv_tr_roc), np.mean(cv_va_roc), roc_te, ic_roc))
        print(format_row("GINI", np.mean(cv_tr_gini), np.mean(cv_va_gini), gini_te, ic_gini))
        print(format_row("PR-AUC", np.mean(cv_tr_pr), np.mean(cv_va_pr), pr_te))
        print(format_row("KS", np.mean(cv_tr_ks), np.mean(cv_va_ks), ks_te))

    if usar_dep:
        pred_test = (prob_test >= threshold).astype(int)
        acc_te = accuracy_score(y_te_arr, pred_test)
        prec_te = precision_score(y_te_arr, pred_test, zero_division=0)
        rec_te = recall_score(y_te_arr, pred_test, zero_division=0)

        print(f"\n [ METRICAS DEPENDENTES DO THRESHOLD (Corte = {threshold:.4f}) ]")
        print(header)
        print("-" * len(header))
        print(format_row("Accuracy", np.mean(cv_tr_acc), np.mean(cv_va_acc), acc_te))
        print(format_row("Precision", np.mean(cv_tr_prec), np.mean(cv_va_prec), prec_te))
        print(format_row("Recall", np.mean(cv_tr_rec), np.mean(cv_va_rec), rec_te))
    print(f"{'='*108}\n")

    return modelo


def avaliar_classificacao_oot(modelo, X_oot, y_oot, X_hist, y_hist, threshold=0.5, nome_modelo="Modelo Campeão"):
    """
    Avalia o modelo na base Out-of-Time (OOT) final, exibindo a proporção de fraude
    e comparando as métricas completas lado a lado com a base Histórica/Teste.
    """
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, accuracy_score,
        precision_score, recall_score, f1_score, confusion_matrix
    )
    from scipy.stats import ks_2samp

    y_hist_arr = np.asarray(y_hist)
    y_oot_arr = np.asarray(y_oot)

    tx_hist = float(np.mean(y_hist_arr)) * 100
    tx_oot = float(np.mean(y_oot_arr)) * 100
    dif_tx = tx_oot - tx_hist

    prob_hist = modelo.predict_proba(X_hist)[:, 1]
    pred_hist = (prob_hist >= threshold).astype(int)
    pos_h, neg_h = prob_hist[y_hist_arr == 1], prob_hist[y_hist_arr == 0]
    met_hist = {
        'ROC-AUC': roc_auc_score(y_hist_arr, prob_hist),
        'PR-AUC': average_precision_score(y_hist_arr, prob_hist),
        'KS': ks_2samp(pos_h, neg_h).statistic if (len(pos_h) > 0 and len(neg_h) > 0) else 0.0,
        'Accuracy': accuracy_score(y_hist_arr, pred_hist),
        'Precision': precision_score(y_hist_arr, pred_hist, zero_division=0),
        'Recall': recall_score(y_hist_arr, pred_hist, zero_division=0),
        'F1-Score': f1_score(y_hist_arr, pred_hist, zero_division=0),
    }

    prob_oot = modelo.predict_proba(X_oot)[:, 1]
    pred_oot = (prob_oot >= threshold).astype(int)
    pos_o, neg_o = prob_oot[y_oot_arr == 1], prob_oot[y_oot_arr == 0]
    met_oot = {
        'ROC-AUC': roc_auc_score(y_oot_arr, prob_oot),
        'PR-AUC': average_precision_score(y_oot_arr, prob_oot),
        'KS': ks_2samp(pos_o, neg_o).statistic if (len(pos_o) > 0 and len(neg_o) > 0) else 0.0,
        'Accuracy': accuracy_score(y_oot_arr, pred_oot),
        'Precision': precision_score(y_oot_arr, pred_oot, zero_division=0),
        'Recall': recall_score(y_oot_arr, pred_oot, zero_division=0),
        'F1-Score': f1_score(y_oot_arr, pred_oot, zero_division=0),
    }

    print("=" * 80)
    print(f" DESEMPENHO OOT FINAL: {nome_modelo.upper()}")
    print(f" Threshold aplicado: {threshold:.4f}")
    print("-" * 80)

    fmt_dif_tx = f"+{dif_tx:.2f}%" if dif_tx >= 0 else f"{dif_tx:.2f}%"
    print(" DISTRIBUIÇÃO DO TARGET (FRAUDE / isFraud)")
    print(f"  -> Histórico/Teste : {tx_hist:.2f}%")
    print(f"  -> OOT Final       : {tx_oot:.2f}% (Variação: {fmt_dif_tx})")
    print("-" * 80)

    print(f"  {'Métrica':<15} | {'Histórico/Teste':<15} | {'OOT Final':<15} | {'Variação (Drop)'}")
    print(f"  {'-'*15} | {'-'*15} | {'-'*15} | {'-'*15}")

    for metrica in met_hist.keys():
        val_h = met_hist[metrica]
        val_o = met_oot[metrica]
        diff = val_o - val_h
        print(f"  {metrica:<15} | {val_h:<15.4f} | {val_o:<15.4f} | {diff:>+8.4f}")

    print("=" * 80)
    print("\nMatriz de Confusão (OOT Final):")
    print(confusion_matrix(y_oot_arr, pred_oot))

    return prob_oot, pred_oot

def comparar_matrizes_cv(modelos_dict, thresholds_dict, X, y, cv=5):
    """
    Aplica Validação Cruzada Estratificada e plota as Matrizes de Confusão
    aplicando um threshold (limiar) de corte customizado para cada modelo.
    """
    num_modelos = len(modelos_dict)
    fig, axes = plt.subplots(nrows=num_modelos, ncols=2, figsize=(14, 5 * num_modelos))
    
    if num_modelos == 1: axes = [axes]
        
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    for i, (nome, modelo) in enumerate(modelos_dict.items()):
        
        # Puxa APENAS as probabilidades
        y_probs = cross_val_predict(modelo, X, y, cv=skf, method='predict_proba', n_jobs=-1)[:, 1]
        
        # Aplica o limiar ótimo descoberto no passo anterior
        t_otimo = thresholds_dict[nome]
        y_pred_customizado = (y_probs >= t_otimo).astype(int)
        
        # 1. Matriz em Valores Absolutos
        ConfusionMatrixDisplay.from_predictions(
            y, y_pred_customizado,
            display_labels=['Não Chove (0)', 'Chove (1)'],
            cmap='Blues', values_format='d', ax=axes[i][0], colorbar=False
        )
        axes[i][0].set_title(f"[{nome}] Absolutos (Threshold={t_otimo:.2f})", fontweight='bold')
        
        # 2. Matriz Normalizada por Recall
        ConfusionMatrixDisplay.from_predictions(
            y, y_pred_customizado,
            display_labels=['Não Chove (0)', 'Chove (1)'],
            cmap='Blues', normalize='true', values_format='.2%', ax=axes[i][1], colorbar=False
        )
        axes[i][1].set_title(f"[{nome}] Taxas Normalizadas (Recall)", fontweight='bold')
        
    plt.tight_layout()
    plt.show()


def gerar_tabela_sazonalidade_oot(modelo, X_hist, y_hist, df_oot, target_col, threshold, col_mes='Month'):
    """
    Gera uma tabela consolidada (DataFrame) comparando a performance 
    do modelo na base Histórica (Baseline) com a base OOT, segregada por período.
    Prioriza as métricas independentes (ROC-AUC, PR-AUC, KS) e a taxa da classe positiva.
    """
    linhas_tabela = []

    # ==========================================
    # 1. LINHA DE REFERÊNCIA (HISTÓRICO)
    # ==========================================
    prob_hist = modelo.predict_proba(X_hist)[:, 1]
    pred_hist = (prob_hist >= threshold).astype(int)
    ks_hist, _ = ks_2samp(prob_hist[y_hist == 1], prob_hist[y_hist == 0])

    linhas_tabela.append({
        'Período': 'Histórico (Baseline)',
        'ROC-AUC': roc_auc_score(y_hist, prob_hist),
        'PR-AUC': average_precision_score(y_hist, prob_hist),
        'KS': ks_hist,
        'Taxa Rain=Yes': y_hist.mean(),
        'Accuracy': accuracy_score(y_hist, pred_hist),
        'Precision': precision_score(y_hist, pred_hist, zero_division=0),
        'Recall': recall_score(y_hist, pred_hist),
        'F1-Score': f1_score(y_hist, pred_hist)
    })

    # ==========================================
    # 2. LINHAS DA BASE OOT (MÊS A MÊS)
    # ==========================================
    numero_meses = df_oot[col_mes].max()

    for mes in range(1, numero_meses + 1):
        filtro = df_oot[col_mes] == mes
        df_mes = df_oot[filtro]
        
        if len(df_mes) == 0:
            continue
            
        X_mes = df_mes.drop(columns=[target_col])
        y_mes = df_mes[target_col]
        
        prob_mes = modelo.predict_proba(X_mes)[:, 1]
        pred_mes = (prob_mes >= threshold).astype(int)
        ks_mes, _ = ks_2samp(prob_mes[y_mes == 1], prob_mes[y_mes == 0])
        
        # Nome formatado (ex: OOT - Mês 01)
        nome_periodo = f'Mês {mes:02d}'
        
        linhas_tabela.append({
            'Período': nome_periodo,
            'ROC-AUC': roc_auc_score(y_mes, prob_mes),
            'PR-AUC': average_precision_score(y_mes, prob_mes),
            'KS': ks_mes,
            'Taxa Rain=Yes': y_mes.mean(),
            'Accuracy': accuracy_score(y_mes, pred_mes),
            'Precision': precision_score(y_mes, pred_mes, zero_division=0),
            'Recall': recall_score(y_mes, pred_mes),
            'F1-Score': f1_score(y_mes, pred_mes)
        })

    # ==========================================
    # 3. GERAÇÃO E FORMATAÇÃO DO DATAFRAME
    # ==========================================
    df_resultados = pd.DataFrame(linhas_tabela)

    # Formata as métricas decimais para 4 casas
    colunas_metricas = ['ROC-AUC', 'PR-AUC', 'KS', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    for col in colunas_metricas:
        df_resultados[col] = df_resultados[col].apply(lambda x: f"{x:.4f}")

    # Formata a Taxa Rain=Yes como percentual
    df_resultados['Taxa Rain=Yes'] = df_resultados['Taxa Rain=Yes'].apply(lambda x: f"{x * 100:.2f}%")

    return df_resultados


def plot_sazonalidade_oot(
    tabela_sazonalidade: pd.DataFrame,
    threshold: float,
    figsize: tuple = (22, 9),
    cor_roc: str = '#0B2545',
    cor_pr: str = '#0096C7',
    cor_recall: str = '#13579B',
    cor_f1: str = '#3E7CC4',
    cor_taxa: str = '#48CAE4',
):
    """
    Plota dois painéis lado a lado com a evolução sazonal das métricas
    do modelo, incluindo a taxa de chuva como eixo secundário.

    Painel esquerdo: capacidade preditiva (ROC-AUC, PR-AUC) — independente do threshold.
    Painel direito: desempenho operacional (Recall, F1-Score) — dependente do threshold.

    Parâmetros
    ----------
    tabela_sazonalidade : pd.DataFrame
        DataFrame retornado por ``gerar_tabela_sazonalidade_oot``.
    threshold : float
        Limiar de corte aplicado (usado no título do painel direito).
    figsize : tuple, default=(22, 9)
        Dimensões da figura.
    cor_roc, cor_pr, cor_recall, cor_f1, cor_taxa : str
        Cores das linhas de cada métrica e da taxa de chuva.
    """
    df_plot = tabela_sazonalidade.copy()

    cols_float = ['ROC-AUC', 'PR-AUC', 'Recall', 'F1-Score']
    for col in cols_float:
        df_plot[col] = df_plot[col].astype(float)
    df_plot['Taxa Rain=Yes'] = df_plot['Taxa Rain=Yes'].str.replace('%', '').astype(float)

    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12})

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor('#f4f4f9')

    # ---- Painel 1: Capacidade Preditiva (Independentes) ----
    ax1.set_facecolor('#ffffff')

    sns.lineplot(data=df_plot, x='Período', y='ROC-AUC', marker='o',
                 color=cor_roc, linewidth=3.5, markersize=12,
                 markeredgecolor='white', markeredgewidth=1.5,
                 label='ROC-AUC', ax=ax1)

    sns.lineplot(data=df_plot, x='Período', y='PR-AUC', marker='s',
                 color=cor_pr, linewidth=3.5, markersize=12,
                 markeredgecolor='white', markeredgewidth=1.5,
                 label='PR-AUC', ax=ax1)

    ax1.axvline(x=0.5, color='gray', linestyle=':', linewidth=2, alpha=0.7)
    ax1.text(0.15, 0.97, 'Baseline (Histórico)', color='gray',
             fontsize=12, fontweight='bold', transform=ax1.get_xaxis_transform())
    ax1.set_title('Capacidade Preditiva Pura\n(Independente do Limiar de Corte)',
                  fontsize=17, fontweight='bold', color='#333', pad=20)
    ax1.set_xlabel('Período Avaliado', fontsize=14, fontweight='bold', color='#555')
    ax1.set_ylabel('Score (0 a 1)', fontsize=14, fontweight='bold', color='#555')
    ax1.set_ylim(0.4, 1.0)
    ax1.tick_params(axis='x', rotation=45, labelsize=12)

    ax2 = ax1.twinx()
    sns.lineplot(data=df_plot, x='Período', y='Taxa Rain=Yes', marker='^',
                 color=cor_taxa, linestyle='--', linewidth=2.5, markersize=12,
                 markeredgecolor='white', markeredgewidth=1.5,
                 label='Taxa de Chuva (%)', ax=ax2)
    ax2.set_ylabel('Taxa de Chuva (%)', fontsize=14, fontweight='bold', color=cor_taxa)
    ax2.set_ylim(0, 45)
    ax2.grid(False)
    ax2.tick_params(axis='y', labelcolor=cor_taxa, labelsize=12)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2,
               loc='lower left', fontsize=12, framealpha=0.9, edgecolor='#ccc', borderpad=1)
    ax2.get_legend().remove()

    # ---- Painel 2: Desempenho Operacional (Dependentes) ----
    ax3.set_facecolor('#ffffff')

    sns.lineplot(data=df_plot, x='Período', y='Recall', marker='D',
                 color=cor_recall, linewidth=3.5, markersize=12,
                 markeredgecolor='white', markeredgewidth=1.5,
                 label='Recall (Acerto de Chuvas)', ax=ax3)

    sns.lineplot(data=df_plot, x='Período', y='F1-Score', marker='X',
                 color=cor_f1, linewidth=3.5, markersize=13,
                 markeredgecolor='white', markeredgewidth=1.5,
                 label='F1-Score', ax=ax3)

    ax3.axvline(x=0.5, color='gray', linestyle=':', linewidth=2, alpha=0.7)
    ax3.text(0.15, 0.97, 'Baseline (Histórico)', color='gray',
             fontsize=12, fontweight='bold', transform=ax3.get_xaxis_transform())
    ax3.set_title(f'Desempenho Operacional\n(Threshold = {threshold:.4f})',
                  fontsize=17, fontweight='bold', color='#333', pad=20)
    ax3.set_xlabel('Período Avaliado', fontsize=14, fontweight='bold', color='#555')
    ax3.set_ylabel('Score (0 a 1)', fontsize=14, fontweight='bold', color='#555')
    ax3.set_ylim(0.4, 1.0)
    ax3.tick_params(axis='x', rotation=45, labelsize=12)

    ax4 = ax3.twinx()
    sns.lineplot(data=df_plot, x='Período', y='Taxa Rain=Yes', marker='^',
                 color=cor_taxa, linestyle='--', linewidth=2.5, markersize=12,
                 markeredgecolor='white', markeredgewidth=1.5,
                 label='Taxa de Chuva (%)', ax=ax4)
    ax4.set_ylabel('Taxa de Chuva (%)', fontsize=14, fontweight='bold', color=cor_taxa)
    ax4.set_ylim(0, 45)
    ax4.grid(False)
    ax4.tick_params(axis='y', labelcolor=cor_taxa, labelsize=12)

    lines_3, labels_3 = ax3.get_legend_handles_labels()
    lines_4, labels_4 = ax4.get_legend_handles_labels()
    ax3.legend(lines_3 + lines_4, labels_3 + labels_4,
               loc='lower left', fontsize=12, framealpha=0.9, edgecolor='#ccc', borderpad=1)
    ax4.get_legend().remove()

    sns.despine(right=False)
    plt.tight_layout(pad=3.0)
    plt.show()


def calcular_psi(esperado, atual, bins=10):
    """
    Calcula o PSI para uma variável numérica contínua.
    esperado: array/série da base Histórica (Treino)
    atual: array/série da base Out-of-Time (Teste/OOT)
    """
    esperado = esperado.dropna()
    atual = atual.dropna()
    
    # 1. Define os limites dos bins baseados nos decis da base histórica
    # Usamos unique() para lidar com variáveis que têm muitos valores repetidos (ex: muitos zeros)
    limites = np.unique(np.percentile(esperado, np.linspace(0, 100, bins + 1)))
    
    # Se a variável tiver muito pouca variação, pode gerar limites de tamanho < 2. Retorna 0.
    if len(limites) < 2:
        return 0.0
    
    # Força os limites extremos ao infinito para capturar outliers na base OOT
    limites[0] = -np.inf
    limites[-1] = np.inf
    
    # 2. Conta quantas observações caem em cada bin
    contagem_esperado, _ = np.histogram(esperado, bins=limites)
    contagem_atual, _ = np.histogram(atual, bins=limites)
    
    # Substitui zeros por um número muito pequeno para evitar erro de divisão/log por zero
    contagem_esperado = np.where(contagem_esperado == 0, 0.001, contagem_esperado)
    contagem_atual = np.where(contagem_atual == 0, 0.001, contagem_atual)
    
    # 3. Converte para proporções (percentuais)
    perc_esperado = contagem_esperado / np.sum(contagem_esperado)
    perc_atual = contagem_atual / np.sum(contagem_atual)
    
    # 4. Fórmula do PSI
    psi = np.sum((perc_atual - perc_esperado) * np.log(perc_atual / perc_esperado))
    
    return psi

def relatorio_feature_drift_psi(X_hist, X_oot, bins=10):
    """
    Varre todas as variáveis numéricas e gera um DataFrame com o PSI,
    o KS de drift e o Status de cada feature.
    """
    cols_num = X_hist.select_dtypes(include=['float64', 'int64']).columns
    resultados = []

    for col in cols_num:
        psi_val = calcular_psi(X_hist[col], X_oot[col], bins=bins)
        ks_val, _ = ks_2samp(
            X_hist[col].dropna(), X_oot[col].dropna()
        )

        if psi_val < 0.10:
            status = " Estável (Sem Drift)"
        elif psi_val < 0.25:
            status = " Alerta (Drift Moderado)"
        else:
            status = " Crítico (Drift Severo)"

        resultados.append({
            'Variável': col,
            'PSI': psi_val,
            'KS': ks_val,
            'Status': status
        })

    df_psi = pd.DataFrame(resultados).sort_values(
        by='PSI', ascending=False
    ).reset_index(drop=True)

    df_psi['PSI'] = df_psi['PSI'].apply(lambda x: f"{x:.4f}")
    df_psi['KS'] = df_psi['KS'].apply(lambda x: f"{x:.4f}")

    return df_psi


def plot_curvas_roc_pr(y_true, y_prob, nome_modelo='', figsize=(14, 6)):
    """
    Plota a curva ROC e a curva Precision-Recall lado a lado.

    Parâmetros
    ----------
    y_true : array-like
        Labels verdadeiros (0/1).
    y_prob : array-like
        Probabilidades da classe positiva.
    nome_modelo : str, default=''
        Nome usado no título dos gráficos.
    figsize : tuple, default=(14, 6)
        Dimensões da figura.
    """
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f'Curvas de Avaliação — {nome_modelo}' if nome_modelo
                 else 'Curvas de Avaliação', fontsize=15, fontweight='bold')

    ax1.plot(fpr, tpr, color='#1f77b4', linewidth=2.5,
             label=f'ROC-AUC = {roc_auc:.4f}')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1, label='Aleatório')
    ax1.fill_between(fpr, tpr, alpha=0.1, color='#1f77b4')
    ax1.set_xlabel('Taxa de Falsos Positivos (1 - Specificity)', fontsize=12)
    ax1.set_ylabel('Taxa de Verdadeiros Positivos (Recall)', fontsize=12)
    ax1.set_title('Curva ROC', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.02])
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(recall, precision, color='#0096C7', linewidth=2.5,
             label=f'PR-AUC = {pr_auc:.4f}')
    prevalencia = y_true.mean() if hasattr(y_true, 'mean') else np.mean(y_true)
    ax2.axhline(y=prevalencia, color='gray', linestyle='--', alpha=0.5,
                label=f'Baseline ({prevalencia:.2%})')
    ax2.fill_between(recall, precision, alpha=0.1, color='#0096C7')
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Curva Precision-Recall', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower left', fontsize=11, framealpha=0.9)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.02])
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_separacao_scores(y_true, y_prob, nome_modelo='', figsize=(12, 5)):
    """
    Visualiza a separação de scores entre positivos e negativos,
    apoiando a interpretação do KS com histogramas sobrepostos e CDFs.

    Parâmetros
    ----------
    y_true : array-like
        Labels verdadeiros (0/1).
    y_prob : array-like
        Probabilidades da classe positiva.
    nome_modelo : str, default=''
        Nome usado no título.
    figsize : tuple, default=(12, 5)
        Dimensões da figura.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    scores_pos = y_prob[y_true == 1]
    scores_neg = y_prob[y_true == 0]
    ks_val, _ = ks_2samp(scores_pos, scores_neg)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f'Separação de Scores — {nome_modelo}' if nome_modelo
                 else 'Separação de Scores', fontsize=15, fontweight='bold')

    bins = np.linspace(0, 1, 60)
    ax1.hist(scores_neg, bins=bins, alpha=0.5, color='#8DB6E3',
             density=True, label='Negativo (Não Chove)')
    ax1.hist(scores_pos, bins=bins, alpha=0.5, color='#0B2545',
             density=True, label='Positivo (Chove)')
    ax1.set_xlabel('Probabilidade Prevista', fontsize=12)
    ax1.set_ylabel('Densidade', fontsize=12)
    ax1.set_title('Distribuição de Scores', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11, framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.5)

    all_sorted = np.sort(y_prob)
    cdf_pos = np.searchsorted(np.sort(scores_pos), all_sorted, side='right') / len(scores_pos)
    cdf_neg = np.searchsorted(np.sort(scores_neg), all_sorted, side='right') / len(scores_neg)

    ax2.plot(all_sorted, cdf_neg, color='#8DB6E3', linewidth=2.5,
             label='CDF Negativo')
    ax2.plot(all_sorted, cdf_pos, color='#0B2545', linewidth=2.5,
             label='CDF Positivo')

    ks_idx = np.argmax(np.abs(cdf_pos - cdf_neg))
    ks_x = all_sorted[ks_idx]
    ax2.plot([ks_x, ks_x],
             [cdf_neg[ks_idx], cdf_pos[ks_idx]],
             color='black', linewidth=2.5, linestyle='--',
             label=f'KS = {ks_val:.4f}')

    ax2.set_xlabel('Probabilidade Prevista', fontsize=12)
    ax2.set_ylabel('CDF Acumulada', fontsize=12)
    ax2.set_title(f'CDFs e Estatística KS ({ks_val:.4f})', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11, framealpha=0.9, loc='center right')
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def tabela_comparacao_final(tabela_sazonalidade, metrics_test, metrics_oot):
    """
    Monta a tabela de comparação final exigida pelo PDF:
    Teste / OOT / Melhor mês / Pior mês com ROC-AUC, PR-AUC e KS.

    Parâmetros
    ----------
    tabela_sazonalidade : pd.DataFrame
        Retornado por ``gerar_tabela_sazonalidade_oot``.
    metrics_test : dict
        Métricas do conjunto de teste com chaves 'roc_auc', 'pr_auc', 'ks'.
    metrics_oot : dict
        Métricas do OOT global com chaves 'roc_auc', 'pr_auc', 'ks'.

    Retorna
    -------
    pd.DataFrame
        Tabela com 4 linhas (Teste, OOT, Melhor mês, Pior mês).
    """
    df_saz = tabela_sazonalidade.copy()
    df_meses = df_saz[df_saz['Período'] != 'Histórico (Baseline)'].copy()

    for col in ['ROC-AUC', 'PR-AUC', 'KS']:
        df_meses[col] = df_meses[col].astype(float)

    melhor_mes = df_meses.loc[df_meses['ROC-AUC'].idxmax()]
    pior_mes = df_meses.loc[df_meses['ROC-AUC'].idxmin()]

    linhas = [
        {
            'Conjunto': 'Teste (Histórico)',
            'ROC-AUC': metrics_test.get('roc_auc', metrics_test.get('ROC-AUC', '-')),
            'PR-AUC': metrics_test.get('pr_auc', metrics_test.get('PR-AUC', '-')),
            'KS': metrics_test.get('ks', metrics_test.get('KS', '-')),
        },
        {
            'Conjunto': 'OOT 2017',
            'ROC-AUC': metrics_oot.get('roc_auc', metrics_oot.get('ROC-AUC', '-')),
            'PR-AUC': metrics_oot.get('pr_auc', metrics_oot.get('PR-AUC', '-')),
            'KS': metrics_oot.get('ks', metrics_oot.get('KS', '-')),
        },
        {
            'Conjunto': f"Melhor mês ({melhor_mes['Período']})",
            'ROC-AUC': f"{melhor_mes['ROC-AUC']:.4f}",
            'PR-AUC': f"{melhor_mes['PR-AUC']:.4f}",
            'KS': f"{melhor_mes['KS']:.4f}",
        },
        {
            'Conjunto': f"Pior mês ({pior_mes['Período']})",
            'ROC-AUC': f"{pior_mes['ROC-AUC']:.4f}",
            'PR-AUC': f"{pior_mes['PR-AUC']:.4f}",
            'KS': f"{pior_mes['KS']:.4f}",
        },
    ]

    return pd.DataFrame(linhas)


def exportar_relatorios(
    tabela_sazonalidade,
    df_drift,
    pasta_reports=None,
):
    """
    Exporta os relatórios de estabilidade temporal e drift como CSV
    na pasta ``reports/``.

    Parâmetros
    ----------
    tabela_sazonalidade : pd.DataFrame
        Tabela de sazonalidade (gerar_tabela_sazonalidade_oot).
    df_drift : pd.DataFrame
        Tabela de drift (relatorio_feature_drift_psi).
    pasta_reports : str, optional
        Caminho da pasta de saída. Se None, usa ``reports/`` relativo
        à raiz do projeto.
    """
    import os as _os

    if pasta_reports is None:
        pasta_reports = _os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)), 'reports'
        )

    _os.makedirs(pasta_reports, exist_ok=True)
    _os.makedirs(_os.path.join(pasta_reports, 'figures'), exist_ok=True)

    caminho_temporal = _os.path.join(pasta_reports, 'temporal_performance.csv')
    tabela_sazonalidade.to_csv(caminho_temporal, index=False)

    caminho_drift = _os.path.join(pasta_reports, 'drift_report.csv')
    df_drift.to_csv(caminho_drift, index=False)

    print(f"  temporal_performance.csv -> {caminho_temporal}")
    print(f"  drift_report.csv        -> {caminho_drift}")


# =============================================================================
# FUNÇÕES DE EDA, FEATURE ENGINEERING, DOWNCAST E MULTIVARIADA (IEEE-CIS / RISCO DE CRÉDITO)
# =============================================================================

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
    plt.show()

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


def aplicar_feature_engineering_tr(df_tr: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica o pipeline de Feature Engineering sobre o DataFrame de transações (`df_train_tr`):
    1. Bloco M: contagem de 'T' e 'F', assinatura M1_M2_M3, M4_limpo e mapeamento numérico (_num).
    2. Cartões: limpeza de categorias raras em card6 e combinação card4 + card6_limpo.
    3. Anomalias no Bloco V: flag is_V14_zero, anomalia em colunas quase-constantes e drop de V305.
    4. Variáveis Temporais, Centavos, Domínios de E-mail e Flags de Nulos/Presença.

    Parâmetros
    ----------
    df_tr : pd.DataFrame
        DataFrame de transações (`df_train_tr`).

    Retorna
    -------
    pd.DataFrame
        DataFrame enriquecido com as novas variáveis.
    """
    # -------------------------------------------------------------------------
    # 1. BLOCO M: FLAGS DE VALIDAÇÃO DE CADASTRO (M1 a M9)
    # -------------------------------------------------------------------------
    cols_m_bool = [c for c in ['M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9'] if c in df_tr.columns]
    if cols_m_bool:
        df_tr['M_true_count'] = (df_tr[cols_m_bool] == 'T').sum(axis=1).astype('int8')
        df_tr['M_false_count'] = (df_tr[cols_m_bool] == 'F').sum(axis=1).astype('int8')

    if all(c in df_tr.columns for c in ['M1', 'M2', 'M3']):
        df_tr['M1_M2_M3_assinatura'] = (
            df_tr['M1'].astype(str).replace('nan', 'X') + '_' +
            df_tr['M2'].astype(str).replace('nan', 'X') + '_' +
            df_tr['M3'].astype(str).replace('nan', 'X')
        )

    if 'M4' in df_tr.columns:
        df_tr['M4_limpo'] = df_tr['M4'].astype(str).replace('nan', 'Missing').astype('category')

    # -------------------------------------------------------------------------
    # 2. COMBINAÇÃO E LIMPEZA DE CARTÃO (card4 e card6)
    # -------------------------------------------------------------------------
    if 'card6' in df_tr.columns:
        df_tr['card6_limpo'] = df_tr['card6'].astype(str).replace({
            'debit or credit': 'debit',
            'charge card': 'credit',
            'nan': 'missing'
        }).fillna('missing')

    if 'card4' in df_tr.columns and 'card6_limpo' in df_tr.columns:
        df_tr['card_brand_type'] = (
            df_tr['card4'].astype(str).replace('nan', 'missing') + '_' + df_tr['card6_limpo'].astype(str)
        ).astype('category')

    # -------------------------------------------------------------------------
    # 3. ANOMALIAS CRÍTICAS E TRATAMENTO NO BLOCO V
    # -------------------------------------------------------------------------
    if 'V14' in df_tr.columns:
        df_tr['is_V14_zero'] = (df_tr['V14'] == 0.0).astype('int8')

    cols_near_const = [c for c in ['V107', 'V117', 'V118', 'V119', 'V120', 'V121', 'V122'] if c in df_tr.columns]
    if cols_near_const:
        df_tr['V_near_constant_anomaly'] = (
            (df_tr[cols_near_const] != 1.0) & df_tr[cols_near_const].notnull()
        ).any(axis=1).astype('int8')

    if 'V305' in df_tr.columns:
        df_tr.drop(columns=['V305'], inplace=True)

    # -------------------------------------------------------------------------
    # 4. MAPEAMENTO NUMÉRICO DO BLOCO M
    # -------------------------------------------------------------------------
    map_tf = {'T': 1, 'F': 0}
    for col in cols_m_bool:
        df_tr[f'{col}_num'] = df_tr[col].map(map_tf)

    # -------------------------------------------------------------------------
    # 5. VARIÁVEIS TEMPORAIS, CENTAVOS, E-MAIL E FLAGS DE PRESENÇA
    # -------------------------------------------------------------------------
    if 'TransactionDT' in df_tr.columns:
        df_tr['Transaction_Hour'] = ((df_tr['TransactionDT'] // 3600) % 24).astype('int8')
        df_tr['Transaction_DayOfWeek'] = ((df_tr['TransactionDT'] // (3600 * 24)) % 7).astype('int8')

    if 'TransactionAmt' in df_tr.columns:
        df_tr['TransactionAmt_Cents'] = np.modf(df_tr['TransactionAmt'])[0].astype('float32')

    if 'P_emaildomain' in df_tr.columns and 'R_emaildomain' in df_tr.columns:
        df_tr['same_email_domain'] = (
            (df_tr['P_emaildomain'] == df_tr['R_emaildomain']) &
            df_tr['P_emaildomain'].notnull()
        ).astype('int8')

    if 'P_emaildomain' in df_tr.columns:
        df_tr['P_email_suffix'] = df_tr['P_emaildomain'].str.split('.').str[-1]

    cols_tr_features = [c for c in df_tr.columns if c not in ['TransactionID', 'isFraud', 'tr_null_count']]
    df_tr['tr_null_count'] = df_tr[cols_tr_features].isnull().sum(axis=1).astype('int16')

    if 'dist2' in df_tr.columns:
        df_tr['has_dist2'] = df_tr['dist2'].notnull().astype('int8')
    if 'R_emaildomain' in df_tr.columns:
        df_tr['has_R_email'] = df_tr['R_emaildomain'].notnull().astype('int8')

    novas_cols = [
        c for c in df_tr.columns
        if any(p in c for p in [
            'M_', 'card_', 'is_V14', 'V_near', '_limpo', '_num',
            'Transaction_Hour', 'Transaction_DayOfWeek', 'TransactionAmt_Cents',
            'same_email_domain', 'P_email_suffix', 'tr_null_count', 'has_dist2', 'has_R_email'
        ])
    ]
    print(f"Feature Engineering concluída em df_train_tr ({len(novas_cols)} features criadas/atualizadas).")
    print(f"Shape atual de df_train_tr: {df_tr.shape}")
    return df_tr


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


def aplicar_feature_engineering_id(df_id: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica o pipeline de Feature Engineering sobre `df_train_id`:
    1. Extração numérica ordinal de `id_34` (`id_34_score`).
    2. Assinatura e contagem de flags de segurança (`id_35` a `id_38`).
    3. Cruzamento de histórico de dispositivo e IP (`id_28` e `id_29`).
    4. Flag de display padrão vs headless/atípico (`id_32`).
    5. Mapeamento numérico de variáveis binárias/ternárias (`_num`).
    6. Limpeza de família de navegador (`id_31`), área de tela (`id_33`) e fabricante (`DeviceInfo`).

    Parâmetros
    ----------
    df_id : pd.DataFrame
        DataFrame de identidade (`df_train_id`).

    Retorna
    -------
    pd.DataFrame
        DataFrame enriquecido com as novas features de identidade.
    """
    # Garante que o tratamento do bloco esparso já foi aplicado
    if 'id_23' in df_id.columns:
        df_id = tratar_nulos_id(df_id)

    # 1. Extração Numérica Ordinal (id_34)
    if 'id_34' in df_id.columns:
        df_id['id_34_score'] = df_id['id_34'].str.split(':').str[1].astype(float)

    # 2. Assinatura e Contagem de Flags de Segurança (id_35 a id_38)
    flags_seguranca = [c for c in ['id_35', 'id_36', 'id_37', 'id_38'] if c in df_id.columns]
    if len(flags_seguranca) == 4:
        df_id['assinatura_seguranca'] = (
            df_id['id_35'].astype(str) + '_' +
            df_id['id_36'].astype(str) + '_' +
            df_id['id_37'].astype(str) + '_' +
            df_id['id_38'].astype(str)
        )
        df_id['contagem_seguranca'] = (df_id[flags_seguranca] == 'T').sum(axis=1).astype('int8')

    # 3. Cruzamento de Histórico e Confiabilidade (id_28 e id_29)
    if 'id_28' in df_id.columns and 'id_29' in df_id.columns:
        df_id['device_ip_consistency'] = (
            df_id['id_28'].fillna('Miss').astype(str) + '_' + df_id['id_29'].fillna('Miss').astype(str)
        )

    # 4. Flag de Dispositivo Atípico/Bots (id_32 - Profundidade de Cores)
    if 'id_32' in df_id.columns:
        df_id['is_standard_display'] = df_id['id_32'].map(
            lambda x: 1 if x in [24.0, 32.0] else (0 if pd.notnull(x) else -1)
        ).astype('int8')

    # 5. Mapeamento Numérico Direto para Modelagem (Variáveis Binárias/Ternárias)
    binary_maps = {
        'id_12': {'NotFound': 0, 'Found': 1},
        'id_15': {'Unknown': 0, 'New': 1, 'Found': 2},
        'id_16': {'NotFound': 0, 'Found': 1},
        'id_28': {'New': 0, 'Found': 1},
        'id_29': {'NotFound': 0, 'Found': 1},
        'id_35': {'F': 0, 'T': 1},
        'id_36': {'F': 0, 'T': 1},
        'id_37': {'F': 0, 'T': 1},
        'id_38': {'F': 0, 'T': 1},
        'DeviceType': {'desktop': 0, 'mobile': 1}
    }
    for col, mapping in binary_maps.items():
        if col in df_id.columns:
            df_id[f'{col}_num'] = df_id[col].map(mapping)

    # 6. Limpeza de Navegadores (id_31)
    if 'id_31' in df_id.columns:
        df_id['browser_family'] = df_id['id_31'].str.split(' ').str[0]
        df_id['browser_family'] = df_id['browser_family'].str.lower().replace({
            'samsung': 'samsung', 'mobile': 'safari', 'google': 'chrome'
        })

    # 7. Extração Matemática da Resolução (id_33)
    if 'id_33' in df_id.columns:
        df_id['screen_width'] = df_id['id_33'].str.split('x').str[0].astype(float)
        df_id['screen_height'] = df_id['id_33'].str.split('x').str[1].astype(float)
        df_id['screen_area'] = df_id['screen_width'] * df_id['screen_height']

    # 8. Padronização de Fabricantes (DeviceInfo)
    if 'DeviceInfo' in df_id.columns:
        df_id['device_brand'] = 'Other'
        df_id.loc[df_id['DeviceInfo'].str.contains('SM-|SAMSUNG', na=False, case=False), 'device_brand'] = 'Samsung'
        df_id.loc[df_id['DeviceInfo'].str.contains('iOS|Mac|Apple', na=False, case=False), 'device_brand'] = 'Apple'
        df_id.loc[df_id['DeviceInfo'].str.contains('Windows', na=False, case=False), 'device_brand'] = 'Windows'
        df_id.loc[df_id['DeviceInfo'].str.contains('Moto|XT', na=False, case=False), 'device_brand'] = 'Motorola'
        df_id.loc[df_id['DeviceInfo'].str.contains('LG-', na=False, case=False), 'device_brand'] = 'LG'
        df_id.loc[df_id['DeviceInfo'].str.contains('Huawei|ALE-|HUAWEI', na=False, case=False), 'device_brand'] = 'Huawei'

    print(f"Feature Engineering concluída em df_train_id. Shape atual: {df_id.shape}")
    return df_id


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


import numpy as np
import pandas as pd

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


def aplicar_ohe(
    df: pd.DataFrame,
    cols: List[str],
    drop_first: bool = True,
) -> pd.DataFrame:
    """
    Aplica One-Hot Encoding às colunas categóricas especificadas,
    substituindo-as pelas dummies no DataFrame.
    """
    cols_existentes = [c for c in cols if c in df.columns]
    if not cols_existentes:
        return df

    df_ohe = pd.get_dummies(
        df[cols_existentes], drop_first=drop_first, dtype='int8'
    )
    df = pd.concat([df.drop(columns=cols_existentes), df_ohe], axis=1)
    print(f"OHE aplicado em {len(cols_existentes)} coluna(s): {cols_existentes}")
    return df


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
        plt.show()

    return df_out, df_relatorio


# ==============================================================================
# FUNÇÕES DE MODELAGEM E OTIMIZAÇÃO (ADICIONADAS DE UTILS_3)
# ==============================================================================

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


def _resolver_taxa_e_groups(taxa_desbalanceamento, groups):
    """
    Identifica de forma defensiva se os argumentos foram passados em ordem invertida
    ou posicional (ex: groups no 5º argumento ao invés de taxa_desbalanceamento).
    """
    if groups is None and isinstance(taxa_desbalanceamento, (np.ndarray, pd.Series, list)):
        return None, taxa_desbalanceamento
    if isinstance(taxa_desbalanceamento, (np.ndarray, pd.Series, list)):
        g = taxa_desbalanceamento
        t = float(groups) if (groups is not None and isinstance(groups, (int, float))) else None
        return t, g
    t = float(taxa_desbalanceamento) if (taxa_desbalanceamento is not None and isinstance(taxa_desbalanceamento, (int, float))) else None
    g = groups if isinstance(groups, (np.ndarray, pd.Series, list)) else None
    return t, g


def objective_lgbm(
    trial,
    X_tune,
    y_tune,
    cv,
    taxa_desbalanceamento: Optional[Union[float, np.ndarray, pd.Series, list]] = None,
    groups: Optional[Union[np.ndarray, pd.Series, list]] = None
):
    """Funcao objetivo Optuna para LightGBM. Maximiza ROC-AUC via CV (suporta GroupKFold via groups)."""
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score

    taxa, grps = _resolver_taxa_e_groups(taxa_desbalanceamento, groups)

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 350),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': -1,
    }
    if taxa is not None:
        params['scale_pos_weight'] = taxa
    else:
        params['class_weight'] = 'balanced'

    modelo = LGBMClassifier(**params)
    auc_scores = []
    splits = cv.split(X_tune, y_tune, groups=grps) if grps is not None else cv.split(X_tune, y_tune)
    for tr_idx, val_idx in splits:
        X_tr = X_tune.iloc[tr_idx] if hasattr(X_tune, 'iloc') else X_tune[tr_idx]
        y_tr = y_tune.iloc[tr_idx] if hasattr(y_tune, 'iloc') else y_tune[tr_idx]
        X_va = X_tune.iloc[val_idx] if hasattr(X_tune, 'iloc') else X_tune[val_idx]
        y_va = y_tune.iloc[val_idx] if hasattr(y_tune, 'iloc') else y_tune[val_idx]

        modelo.fit(X_tr, y_tr)
        probs = modelo.predict_proba(X_va)[:, 1]
        auc_scores.append(roc_auc_score(np.asarray(y_va), probs))
    return float(np.mean(auc_scores))


def objective_xgb(
    trial,
    X_tune,
    y_tune,
    cv,
    taxa_desbalanceamento: Union[float, np.ndarray, pd.Series, list] = 1.0,
    groups: Optional[Union[np.ndarray, pd.Series, list]] = None
):
    """Funcao objetivo Optuna para XGBoost. Maximiza ROC-AUC via CV (suporta GroupKFold via groups)."""
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score

    taxa, grps = _resolver_taxa_e_groups(taxa_desbalanceamento, groups)
    if taxa is None:
        taxa = 1.0

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 350),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10.0, log=True),
        'scale_pos_weight': taxa,
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'n_jobs': -1,
        'random_state': 42,
        'verbosity': 0,
    }

    modelo = XGBClassifier(**params)
    auc_scores = []
    splits = cv.split(X_tune, y_tune, groups=grps) if grps is not None else cv.split(X_tune, y_tune)
    for tr_idx, val_idx in splits:
        X_tr = X_tune.iloc[tr_idx] if hasattr(X_tune, 'iloc') else X_tune[tr_idx]
        y_tr = y_tune.iloc[tr_idx] if hasattr(y_tune, 'iloc') else y_tune[tr_idx]
        X_va = X_tune.iloc[val_idx] if hasattr(X_tune, 'iloc') else X_tune[val_idx]
        y_va = y_tune.iloc[val_idx] if hasattr(y_tune, 'iloc') else y_tune[val_idx]

        modelo.fit(X_tr, y_tr, verbose=False)
        probs = modelo.predict_proba(X_va)[:, 1]
        auc_scores.append(roc_auc_score(np.asarray(y_va), probs))
    return float(np.mean(auc_scores))


def objective_cat(
    trial,
    X_tune,
    y_tune,
    cv,
    taxa_desbalanceamento: Optional[Union[float, np.ndarray, pd.Series, list]] = None,
    groups: Optional[Union[np.ndarray, pd.Series, list]] = None
):
    """Funcao objetivo Optuna para CatBoost. Maximiza ROC-AUC via CV (suporta GroupKFold via groups)."""
    from catboost import CatBoostClassifier
    from sklearn.metrics import roc_auc_score

    taxa, grps = _resolver_taxa_e_groups(taxa_desbalanceamento, groups)

    params = {
        'iterations': trial.suggest_int('iterations', 100, 350),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 3, 8),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'eval_metric': 'AUC',
        'random_seed': 42,
        'verbose': False,
        'thread_count': -1,
    }
    if taxa is not None:
        params['scale_pos_weight'] = taxa
    else:
        params['auto_class_weights'] = 'Balanced'

    modelo = CatBoostClassifier(**params)
    auc_scores = []
    splits = cv.split(X_tune, y_tune, groups=grps) if grps is not None else cv.split(X_tune, y_tune)
    for tr_idx, val_idx in splits:
        X_tr = X_tune.iloc[tr_idx] if hasattr(X_tune, 'iloc') else X_tune[tr_idx]
        y_tr = y_tune.iloc[tr_idx] if hasattr(y_tune, 'iloc') else y_tune[tr_idx]
        X_va = X_tune.iloc[val_idx] if hasattr(X_tune, 'iloc') else X_tune[val_idx]
        y_va = y_tune.iloc[val_idx] if hasattr(y_tune, 'iloc') else y_tune[val_idx]

        modelo.fit(X_tr, y_tr)
        probs = modelo.predict_proba(X_va)[:, 1]
        auc_scores.append(roc_auc_score(np.asarray(y_va), probs))
    return float(np.mean(auc_scores))


# Alias de conveniência
objective_lgbm_grouped = objective_lgbm
objective_xgb_grouped = objective_xgb
objective_cat_grouped = objective_cat

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

def extrair_metricas_avaliacao(modelo, X, y, threshold=0.5):
    """
    Calcula dicionário com métricas completas de avaliação em uma partição:
    accuracy, precision, recall, f1_score, roc_auc, pr_auc e ks.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score
    )
    from scipy.stats import ks_2samp

    y_arr = np.asarray(y)
    probs = modelo.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    pos_probs = probs[y_arr == 1]
    neg_probs = probs[y_arr == 0]
    ks_val = float(ks_2samp(pos_probs, neg_probs).statistic) if len(pos_probs) > 0 and len(neg_probs) > 0 else 0.0

    metricas = {
        'threshold': float(threshold),
        'accuracy': float(accuracy_score(y_arr, preds)),
        'precision': float(precision_score(y_arr, preds, zero_division=0)),
        'recall': float(recall_score(y_arr, preds, zero_division=0)),
        'f1_score': float(f1_score(y_arr, preds, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_arr, probs)),
        'pr_auc': float(average_precision_score(y_arr, probs)),
        'ks': float(ks_val)
    }
    return metricas


def exportar_artefatos_modelo(
    nome_modelo,
    modelo,
    diretorio_destino,
    threshold_otimo,
    features_utilizadas,
    X_train,
    y_train,
    X_test,
    y_test,
    X_oot=None,
    y_oot=None,
):
    """
    Exporta o modelo em .joblib e grava os arquivos estruturados
    metadata.json e metrics.json (incluindo OOT quando fornecido).
    """
    import json
    from pathlib import Path
    try:
        import joblib
    except ImportError:
        print("[AVISO] 'joblib' não está instalado. Execute: pip install joblib")
        return

    dest = Path(diretorio_destino)
    dest.mkdir(parents=True, exist_ok=True)

    # 1. Serialização do estimador
    arquivo_modelo = dest / f"{nome_modelo}.joblib"
    joblib.dump(modelo, arquivo_modelo)

    # 2. Extração dos hiperparâmetros
    hiperparametros = {}
    if hasattr(modelo, 'get_params'):
        try:
            params_raw = modelo.get_params(deep=False)
            for k, v in params_raw.items():
                if isinstance(v, (np.integer,)):
                    hiperparametros[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    hiperparametros[k] = float(v)
                elif isinstance(v, (int, float, str, bool, list, dict)) or v is None:
                    hiperparametros[k] = v
                else:
                    hiperparametros[k] = str(v)
        except Exception:
            hiperparametros = {'info': 'Nao foi possivel serializar get_params()'}

    # 3. Geração do metadata.json
    metadata = {
        'nome_modelo': nome_modelo,
        'classe': f"{modelo.__class__.__module__}.{modelo.__class__.__name__}",
        'hiperparametros': hiperparametros,
        'features_utilizadas': list(features_utilizadas),
        'total_features': len(features_utilizadas),
        'tamanho_historico_treino': {
            'linhas': int(len(X_train)),
            'taxa_alvo': float(np.mean(y_train))
        },
        'tamanho_historico_teste': {
            'linhas': int(len(X_test)),
            'taxa_alvo': float(np.mean(y_test))
        }
    }
    if X_oot is not None and y_oot is not None:
        metadata['tamanho_historico_oot'] = {
            'linhas': int(len(X_oot)),
            'taxa_alvo': float(np.mean(y_oot))
        }

    arquivo_metadata = dest / f"{nome_modelo}_metadata.json"
    with open(arquivo_metadata, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # 4. Geração do metrics.json
    metricas_treino = extrair_metricas_avaliacao(modelo, X_train, y_train, threshold=threshold_otimo)
    metricas_teste  = extrair_metricas_avaliacao(modelo, X_test, y_test, threshold=threshold_otimo)

    metrics_payload = {
        'nome_modelo': nome_modelo,
        'threshold_otimo': float(threshold_otimo),
        'metricas_treino': metricas_treino,
        'metricas_teste': metricas_teste
    }
    if X_oot is not None and y_oot is not None:
        metrics_payload['metricas_oot'] = extrair_metricas_avaliacao(modelo, X_oot, y_oot, threshold=threshold_otimo)

    arquivo_metrics = dest / f"{nome_modelo}_metrics.json"
    with open(arquivo_metrics, 'w', encoding='utf-8') as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    print(f"Artefatos de {nome_modelo} exportados com sucesso em {dest}:")
    print(f" - {arquivo_modelo.name}")
    print(f" - {arquivo_metadata.name}")
    print(f" - {arquivo_metrics.name}")


# ==============================================================================
# PODA DE FEATURES (CURVA SHAP), OPTUNA DOS BOOSTINGS E ANÁLISE DE THRESHOLD
# ==============================================================================

_PALETA_POLITICAS = ['#97A3B0', '#5DADE2', '#0096C7', '#13579B', '#0B2545']


def _limpar_bordas(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


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
    plt.show()

    # 5. Listas de Features exportadas
    features_max = list(ranking_features[:n_max])
    features_otimizadas = list(ranking_features[:n_otimizado])

    return df_curva, df_resumo, features_max, features_otimizadas


def plot_optuna_boostings(studies: Dict[str, Any], figsize: tuple = (16, 5)):
    """
    Painel analítico comparando os estudos Optuna dos boostings:
      (1) convergência — ROC-AUC de cada trial e melhor valor acumulado;
      (2) dispersão do ROC-AUC entre trials (sensibilidade aos hiperparâmetros).
    """
    cores = ['#0B2545', '#0096C7', '#5DADE2']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [2, 1]})

    dados_box, nomes = [], []
    for (nome, study), cor in zip(studies.items(), cores):
        valores = np.array([t.value for t in study.trials if t.value is not None])
        trials = np.arange(1, len(valores) + 1)
        ax1.scatter(trials, valores, color=cor, alpha=0.35, s=22)
        ax1.plot(trials, np.maximum.accumulate(valores), color=cor, lw=2.2,
                 label=f'{nome} (melhor = {valores.max():.4f})')
        dados_box.append(valores)
        nomes.append(nome)

    ax1.set_title('Convergência do Optuna — ROC-AUC por Trial (GroupKFold)', fontsize=12, fontweight='bold', loc='left')
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('ROC-AUC médio (CV)')
    ax1.grid(axis='y', linestyle=':', alpha=0.5)
    ax1.legend(frameon=False, loc='lower right')

    bp = ax2.boxplot(dados_box, patch_artist=True, widths=0.5)
    ax2.set_xticks(range(1, len(nomes) + 1))
    ax2.set_xticklabels(nomes)
    for patch, cor in zip(bp['boxes'], cores):
        patch.set_facecolor(cor)
        patch.set_alpha(0.35)
    for i, valores in enumerate(dados_box, start=1):
        ax2.scatter(i, valores.max(), marker='*', s=160, color='#0096C7', edgecolor='#0B2545', linewidth=0.8, zorder=3)
    ax2.set_title('Dispersão entre Trials (★ = melhor)', fontsize=12, fontweight='bold', loc='left')
    ax2.set_ylabel('ROC-AUC médio (CV)')
    ax2.grid(axis='y', linestyle=':', alpha=0.5)

    for ax in (ax1, ax2):
        _limpar_bordas(ax)
    plt.tight_layout()
    plt.show()

    resumo = pd.DataFrame({
        'Modelo': nomes,
        'Melhor_AUC': [v.max() for v in dados_box],
        'Mediana_AUC': [np.median(v) for v in dados_box],
        'Desvio_AUC': [v.std() for v in dados_box],
        'Trials': [len(v) for v in dados_box],
    }).sort_values('Melhor_AUC', ascending=False).reset_index(drop=True)
    return resumo


def gerar_probs_calibradas(classe_modelo, params: dict, X_train, y_train, X_val, groups, n_splits: int = 4):
    """
    1. Gera probabilidades Out-of-Fold no Treino via GroupKFold temporal (sem leakage);
    2. Treina o modelo em 100% do Treino e prediz a Validação;
    3. Ajusta a Calibração Isotônica APENAS no OOF e aplica no OOF e na Validação.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import GroupKFold

    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    oof_raw = np.zeros(len(X_train))
    for tr_idx, va_idx in gkf.split(X_train, y_train, groups=groups):
        m = classe_modelo(**params)
        m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        oof_raw[va_idx] = m.predict_proba(X_train.iloc[va_idx])[:, 1]

    modelo_full = classe_modelo(**params)
    modelo_full.fit(X_train, y_train)
    val_raw = modelo_full.predict_proba(X_val)[:, 1]

    calibrador = IsotonicRegression(out_of_bounds='clip')
    oof_cal = calibrador.fit_transform(oof_raw, y_train)
    val_cal = calibrador.transform(val_raw)

    return {
        'oof_raw': oof_raw, 'oof_cal': oof_cal,
        'val_raw': val_raw, 'val_cal': val_cal,
        'modelo_full': modelo_full, 'calibrador': calibrador,
    }


def threshold_calibrado_para_bruto(calibrador, threshold_cal: float) -> float:
    """
    Converte um corte no espaço CALIBRADO (isotônico) para o corte equivalente na
    probabilidade BRUTA do modelo (a isotônica é monotônica não decrescente).
    """
    xs = np.asarray(calibrador.X_thresholds_, dtype=float)
    ys = np.asarray(calibrador.y_thresholds_, dtype=float)
    idx = int(np.argmax(ys >= threshold_cal))
    if ys[idx] < threshold_cal:
        return float(xs[-1] + 1e-12)
    if idx == 0 or ys[idx] == ys[idx - 1]:
        return float(xs[idx])
    frac = (threshold_cal - ys[idx - 1]) / (ys[idx] - ys[idx - 1])
    return float(xs[idx - 1] + frac * (xs[idx] - xs[idx - 1]))


def obter_amt_real(df: pd.DataFrame, col: str = 'TransactionAmt') -> np.ndarray:
    """Retorna TransactionAmt em US$ reais (reverte log1p se a coluna estiver em escala log)."""
    valores = df[col].values.astype(float)
    return np.expm1(valores) if df[col].mean() < 10 else valores


def blocos_mensais(df: pd.DataFrame, col_dt: str = 'TransactionDT') -> np.ndarray:
    """Índice do bloco mensal (30 dias) de cada transação."""
    return (df[col_dt] // (3600 * 24 * 30)).astype(int).values


def calcular_score_financeiro(y_true, y_pred_block, amt, exibir_detalhes=False, titulo=""):
    """
    Score financeiro conforme regra de negócio (em US$ reais):
      VP = Fraude evitada (0 custo direto) | VN = +0.02 * Amt
      FP = -0.02 * Amt (receita perdida)  | FN = -1.00 * Amt (chargeback)
    """
    y_true = np.asarray(y_true)
    amt = np.asarray(amt, dtype=float)
    bloqueado = np.asarray(y_pred_block, dtype=bool)
    aprovado = ~bloqueado

    vp = bloqueado & (y_true == 1)
    vn = aprovado  & (y_true == 0)
    fp = bloqueado & (y_true == 0)
    fn = aprovado  & (y_true == 1)

    qtd_vp, qtd_vn = int(vp.sum()), int(vn.sum())
    qtd_fp, qtd_fn = int(fp.sum()), int(fn.sum())

    vol_vp = float(amt[vp].sum())
    vol_vn = float(amt[vn].sum())
    vol_fp = float(amt[fp].sum())
    vol_fn = float(amt[fn].sum())

    receita_vn  = 0.02 * vol_vn
    penal_fp    = 0.02 * vol_fp
    prejuizo_fn = 1.00 * vol_fn
    perda_total = penal_fp + prejuizo_fn
    resultado_liquido = receita_vn - perda_total

    approval_rate = qtd_vn / max(1, (y_true == 0).sum())
    recall = qtd_vp / max(1, (y_true == 1).sum())
    precision = qtd_vp / max(1, bloqueado.sum())

    resultado = {
        'Resultado_Liquido': resultado_liquido,
        'Receita_VN': receita_vn,
        'Penalidade_FP': penal_fp,
        'Prejuizo_FN': prejuizo_fn,
        'Perda_Total_FP_FN': perda_total,
        'Fraude_Evitada_VP': vol_vp,
        'Volume_VN_USD': vol_vn,
        'Volume_FP_USD': vol_fp,
        'Qtd_VP': qtd_vp,
        'Qtd_VN': qtd_vn,
        'Qtd_FP': qtd_fp,
        'Qtd_FN': qtd_fn,
        'Approval_Rate': approval_rate,
        'Recall': recall,
        'Precision': precision,
    }

    if exibir_detalhes:
        cabecalho = f"DETALHAMENTO FINANCEIRO — {titulo}" if titulo else "DETALHAMENTO FINANCEIRO"
        print("=" * 88)
        print(cabecalho)
        print("-" * 88)
        print(f"[+] GANHO  (VN — Legítimas Aprovadas | +2% MDR): "
              f"{qtd_vn:>7,} tx | Vol: $ {vol_vn:>14,.2f} | Receita: +$ {receita_vn:>12,.2f}")
        print(f"[*] EVITADO (VP — Fraudes Bloqueadas |  0% MDR): "
              f"{qtd_vp:>7,} tx | Fraude Evitada:             $ {vol_vp:>12,.2f}")
        print(f"[-] PERDA FP (Clientes Bons Bloqueados | -2% MDR): "
              f"{qtd_fp:>7,} tx | Vol: $ {vol_fp:>14,.2f} | Perda:   -$ {penal_fp:>12,.2f}")
        print(f"[-] PERDA FN (Fraudes Não Detectadas   | -100%):  "
              f"{qtd_fn:>7,} tx | Chargeback:                 -$ {prejuizo_fn:>12,.2f}")
        print("-" * 88)
        print(f"Perda Total (FP + FN): -$ {perda_total:>12,.2f} | "
              f"RESULTADO LÍQUIDO: $ {resultado_liquido:>14,.2f}")
        print(f"Aprovação (VN): {approval_rate*100:.2f}% | Recall (VP): {recall*100:.2f}% | Precision: {precision*100:.2f}%")
        print("=" * 88)

    return resultado


# ------------------------------------------------------------------------------
# THRESHOLD FIXO (SEÇÃO 7.2): BUSCA NO TREINO E RESULTADO TOTAL POR BASE
# ------------------------------------------------------------------------------
_CORES_MODELOS = {
    'LGBM_Max':       {'solida': '#0077B6', 'media': '#00B4D8', 'clara': '#90E0EF'},
    'LGBM_Otimizado': {'solida': '#0B2545', 'media': '#13579B', 'clara': '#8DB6E3'},
}
_CORES_EXTRAS = [
    {'solida': '#1D3557', 'media': '#457B9D', 'clara': '#A8DADC'},
    {'solida': '#023E8A', 'media': '#0096C7', 'clara': '#ADE8F4'},
]


def _cor_modelo(nome: str, idx: int = 0) -> Dict[str, str]:
    return _CORES_MODELOS.get(nome, _CORES_EXTRAS[idx % len(_CORES_EXTRAS)])


def threshold_equilibrio(mdr: float = 0.02, perda_fraude: float = 1.00) -> float:
    """
    Corte teórico de equilíbrio para uma probabilidade CALIBRADA p:
      Aprovar  -> (1 - p) * (+mdr) + p * (-perda_fraude)
      Bloquear -> (1 - p) * (-mdr) + p * 0
    Bloquear compensa quando p * perda_fraude > (1 - p) * 2 * mdr, ou seja,
    p* = 2·mdr / (perda_fraude + 2·mdr). O valor da transação se cancela.
    """
    return 2 * mdr / (perda_fraude + 2 * mdr)


def curva_lucro_threshold(y_true, probs, amt, thresholds) -> pd.DataFrame:
    """
    Lucro líquido e métricas operacionais para cada threshold fixo (bloqueia se prob >= threshold).
    Implementação vetorizada (ordenação + somas acumuladas): exata e rápida para milhares de cortes.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(probs, dtype=float)
    a = np.asarray(amt, dtype=float)
    t = np.asarray(thresholds, dtype=float)

    ordem = np.argsort(p, kind='mergesort')
    p_s, y_s, a_s = p[ordem], y[ordem], a[ordem]

    v_aprovar = np.where(y_s == 0, 0.02 * a_s, -a_s)    # VN = +2% | FN = -100%
    v_bloquear = np.where(y_s == 0, -0.02 * a_s, 0.0)   # FP = -2%  | VP = 0

    def _sufixo(v):
        return np.concatenate([np.cumsum(v[::-1])[::-1], [0.0]])

    suf_ganho_bloqueio = _sufixo(v_bloquear - v_aprovar)
    suf_fraude = _sufixo(y_s.astype(float))
    suf_legit = _sufixo((1 - y_s).astype(float))

    idx = np.searchsorted(p_s, t, side='left')           # primeiro índice com prob >= threshold
    n_fraude, n_legit = max(1, int(y.sum())), max(1, int((y == 0).sum()))
    fraudes_bloq, legit_bloq = suf_fraude[idx], suf_legit[idx]
    n_bloq = fraudes_bloq + legit_bloq

    return pd.DataFrame({
        'Threshold': t,
        'Lucro': v_aprovar.sum() + suf_ganho_bloqueio[idx],
        'Aprovacao': 1 - legit_bloq / n_legit,
        'Recall': fraudes_bloq / n_fraude,
        'Precision': np.divide(fraudes_bloq, n_bloq, out=np.zeros_like(n_bloq), where=n_bloq > 0),
    })


def otimizar_threshold_fixo(y_true, probs, amt, t_max: float = 0.50) -> Dict[str, float]:
    """
    Busca EXATA do threshold fixo de maior lucro: avalia todos os valores distintos de probabilidade
    (cada um é um corte diferente). O threshold retornado é o ponto médio entre o nível vencedor e o
    nível imediatamente abaixo: mesmo lucro na base de busca, com margem máxima para bases novas.
    """
    p = np.asarray(probs, dtype=float)
    niveis = np.unique(p)
    niveis = niveis[(niveis > 0) & (niveis <= t_max)]
    curva = curva_lucro_threshold(y_true, p, amt, niveis)
    i = int(curva['Lucro'].values.argmax())
    anterior = niveis[i - 1] if i > 0 else 0.0
    melhor = curva.iloc[i]
    return {
        'Threshold': float((niveis[i] + anterior) / 2),
        'Lucro': float(melhor['Lucro']),
        'Aprovacao': float(melhor['Aprovacao']),
        'Recall': float(melhor['Recall']),
        'Precision': float(melhor['Precision']),
    }


def tabela_varredura_threshold_fixo(
    threshold_otimo: Dict[str, float], curvas: Dict[str, pd.DataFrame], pct_plato: float = 0.95,
    p_equilibrio: Optional[float] = None,
) -> pd.DataFrame:
    """Resumo da busca no OOF do Treino: corte ótimo, lucro máximo, platô e taxas operacionais no pico."""
    linhas = []
    for nome, cv in curvas.items():
        th = threshold_otimo[nome]
        x = cv['Threshold'].values
        lucro_max = float(np.interp(th, x, cv['Lucro'].values))
        aprov_ot  = float(np.interp(th, x, cv['Aprovacao'].values))
        rec_ot    = float(np.interp(th, x, cv['Recall'].values))
        prec_ot   = float(np.interp(th, x, cv['Precision'].values))
        plato = cv.loc[cv['Lucro'] >= pct_plato * lucro_max, 'Threshold']
        linha = {
            'Modelo': nome,
            'Corte Ótimo': th,
            f'Platô ≥{pct_plato:.0%} do Lucro': f"{plato.min():.2%} – {plato.max():.2%}",
            'Lucro Máximo (Treino OOF)': lucro_max,
            'Aprovação no Ótimo': aprov_ot,
            'Recall no Ótimo': rec_ot,
            'Precision no Ótimo': prec_ot,
        }
        if p_equilibrio is not None:
            linha['Equilíbrio Teórico'] = p_equilibrio
        linhas.append(linha)
    return pd.DataFrame(linhas).set_index('Modelo')


def _estilo_limpo(ax):
    """Estética padrão dos gráficos da Seção 7: grid horizontal suave, só a borda inferior, sem ticks."""
    ax.grid(axis='y', ls='-', lw=0.6, color='#EAEAEA')
    ax.grid(axis='x', visible=False)
    ax.set_axisbelow(True)
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.tick_params(axis='both', colors='#555555', length=0, pad=5)


def _desenhar_curva_lucro_ax(
    ax, curva: pd.DataFrame, threshold_otimo: float, nome_modelo: str, idx_cor: int = 0,
    p_equilibrio: Optional[float] = None, pct_plato: float = 0.95, y_limits: Optional[tuple] = None,
):
    """Renderiza uma curva de lucro limpa e minimalista (sem caixas/setas poluindo o gráfico)."""
    cores = _cor_modelo(nome_modelo, idx_cor)
    cor = cores['solida']
    x = curva['Threshold'].values
    y = curva['Lucro'].values / 1e3
    lucro_ot = float(np.interp(threshold_otimo, x, y))

    if y_limits is not None:
        ax.set_ylim(*y_limits)
    else:
        y_lo, y_hi = min(0.0, float(y.min())), float(y.max())
        span = max(1.0, y_hi - y_lo)
        ax.set_ylim(y_lo - 0.08 * span, y_hi + 0.18 * span)

    ax.set_xlim(float(x.min()), float(x.max()))

    # Platô sombreado discreto (informado na legenda)
    if lucro_ot > 0:
        msk = y >= pct_plato * lucro_ot
        if msk.any():
            p_min, p_max = float(x[msk].min()), float(x[msk].max())
            ax.axvspan(p_min, p_max, color=cor, alpha=0.08, lw=0,
                       label=f'Platô ≥{pct_plato:.0%} ({p_min:.1%} – {p_max:.1%})')

    # Linha zero e curva principal
    ax.axhline(0, color='#1A1A1A', lw=0.9)
    ax.fill_between(x, y, 0, where=(y >= 0), interpolate=True, color=cor, alpha=0.05, lw=0)
    ax.plot(x, y, color=cor, lw=2.6, label='Lucro Líquido (OOF)')

    # Equilíbrio teórico (apenas linha tracejada discreta + legenda)
    if p_equilibrio is not None:
        ax.axvline(p_equilibrio, color='#777777', ls=':', lw=1.5,
                   label=f'Equilíbrio Teórico ({p_equilibrio:.2%})')

    # Ponto ótimo: marcador + rótulo curto direto acima do pico
    ax.vlines(threshold_otimo, 0, lucro_ot, color=cor, linestyle='--', lw=1.3, alpha=0.7)
    ax.scatter([threshold_otimo], [lucro_ot], color=cor, s=90, edgecolor='white', linewidth=1.8, zorder=5,
               label=f'Corte Ótimo: {threshold_otimo:.2%} (US$ {lucro_ot:,.1f}k)')
    ax.annotate(
        f"{threshold_otimo:.2%}  (US$ {lucro_ot:,.1f} mil)",
        xy=(threshold_otimo, lucro_ot), xytext=(0, 10), textcoords='offset points',
        ha='center', va='bottom', fontsize=10, fontweight='bold', color=cor,
    )

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.02))
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('US$ {x:,.0f}k'))
    ax.set_xlabel('Threshold de bloqueio (probabilidade calibrada)', fontsize=11, color='#333333', labelpad=8)
    ax.set_ylabel('Lucro líquido (US$ mil)', fontsize=11, color='#333333', labelpad=8)
    ax.set_title(f'Lucro por Threshold Fixo — {nome_modelo} (Treino OOF)',
                 fontsize=13, fontweight='heavy', loc='left', pad=12, color='#1A1A1A')
    ax.legend(frameon=False, fontsize=9.5, loc='upper right')
    _estilo_limpo(ax)


def plot_curva_lucro_threshold(
    curva, threshold_otimo, nome_modelo: str = '',
    p_equilibrio: Optional[float] = None, pct_plato: float = 0.95,
    titulo: str = '', figsize: Optional[tuple] = None,
):
    """
    Curva limpa de lucro líquido (US$ mil) por threshold fixo.
    Aceita um único DataFrame (um modelo por célula) ou um dict de DataFrames.
    """
    if isinstance(curva, dict):
        nomes = list(curva.keys())
        n_mod = len(nomes)
        figsize = figsize or (7.5 * n_mod, 4.8)
        fig, axes = plt.subplots(1, n_mod, figsize=figsize, sharey=True)
        if n_mod == 1:
            axes = [axes]

        all_y = np.concatenate([df_c['Lucro'].values / 1e3 for df_c in curva.values()])
        y_lo, y_hi = min(0.0, float(all_y.min())), float(all_y.max())
        span = max(1.0, y_hi - y_lo)
        y_limits = (y_lo - 0.08 * span, y_hi + 0.18 * span)

        for i, (ax, nome) in enumerate(zip(axes, nomes)):
            th_m = threshold_otimo[nome] if isinstance(threshold_otimo, dict) else float(threshold_otimo)
            _desenhar_curva_lucro_ax(
                ax, curva[nome], th_m, nome, idx_cor=i,
                p_equilibrio=p_equilibrio, pct_plato=pct_plato, y_limits=y_limits,
            )
            if i > 0:
                ax.set_ylabel('')
                ax.tick_params(labelleft=True)

        if titulo:
            fig.suptitle(titulo, fontsize=13.5, fontweight='heavy', x=0.01, ha='left', y=1.02, color='#1A1A1A')
        plt.tight_layout()
        plt.show()
        return

    figsize = figsize or (11, 4.8)
    fig, ax = plt.subplots(figsize=figsize)
    _desenhar_curva_lucro_ax(
        ax, curva, float(threshold_otimo), nome_modelo or 'Modelo',
        p_equilibrio=p_equilibrio, pct_plato=pct_plato,
    )
    if titulo:
        ax.set_title(titulo, fontsize=13, fontweight='heavy', loc='left', pad=12, color='#1A1A1A')
    plt.tight_layout()
    plt.show()


_ROTULO_CURTO_POLITICA = {
    'Sem Modelo': 'sem modelo',
    'Clássico 0.50': '0.50',
    'Fixo Ótimo': 'fixo',
    'Dinâmico (Optuna)': 'dinâmico',
}


def politicas_threshold(prob_raw, prob_cal, threshold_fixo: float, threshold_dinamico_arr=None) -> Dict[str, np.ndarray]:
    """Máscaras de bloqueio de cada política (True = bloqueia)."""
    prob_raw, prob_cal = np.asarray(prob_raw), np.asarray(prob_cal)
    politicas = {
        'Sem Modelo': np.zeros(len(prob_raw), dtype=bool),
        'Clássico 0.50': prob_raw >= 0.50,
        'Fixo Ótimo': prob_cal >= threshold_fixo,
    }
    if threshold_dinamico_arr is not None:
        politicas['Dinâmico (Optuna)'] = prob_cal >= np.asarray(threshold_dinamico_arr)
    return politicas


def resumo_politicas(y_true, amt, politicas_por_modelo: Dict[str, Dict[str, np.ndarray]], base: str) -> pd.DataFrame:
    """DRE total de cada política, por modelo, em uma base (índice: Base, Modelo, Política)."""
    linhas = []
    for nome, politicas in politicas_por_modelo.items():
        for pol, bloqueio in politicas.items():
            r = calcular_score_financeiro(y_true, bloqueio, amt)
            linhas.append({
                'Base': base, 'Modelo': nome, 'Política': pol,
                'Lucro Líquido': r['Resultado_Liquido'],
                'Receita VN (+2%)': r['Receita_VN'],
                'Perda FP (−2%)': r['Penalidade_FP'],
                'Chargeback FN (−100%)': r['Prejuizo_FN'],
                'Aceitação Geral': 1 - (r['Qtd_VP'] + r['Qtd_FP']) / max(1, len(bloqueio)),
                'Aprovação': r['Approval_Rate'],
                'Recall': r['Recall'],
                'Precision': r['Precision'],
            })
    return pd.DataFrame(linhas).set_index(['Base', 'Modelo', 'Política'])


def estilizar_resumo_politicas(df_resumo: pd.DataFrame, politica_destaque: str = 'Fixo Ótimo'):
    """
    Styler de alto contraste (compatível com tema escuro e claro do VS Code/Colab) para `resumo_politicas`
    e `tabela_consolidada_treino_validacao`. Define cores explícitas de fundo e de fonte em todas as células.
    """
    fundos_normais = {
        'Treino (OOF)': '#F5F9FD',
        'Validação': '#EBF5FB',
        'Treino + Validação (Unido)': '#E3F2FD',
    }
    fundos_destaque = {
        'Treino (OOF)': '#D6EAF8',
        'Validação': '#AED6F1',
        'Treino + Validação (Unido)': '#90CAF9',
    }
    bases = list(dict.fromkeys(df_resumo.index.get_level_values('Base')))
    primeiras_linhas = {df_resumo.index[df_resumo.index.get_level_values('Base') == b][0] for b in bases}

    modelos_por_base = {
        (b, m): df_resumo.index[(df_resumo.index.get_level_values('Base') == b) & (df_resumo.index.get_level_values('Modelo') == m)][0]
        for b in bases
        for m in dict.fromkeys(df_resumo.loc[b].index.get_level_values('Modelo'))
    }
    primeiras_linhas_modelo = set(modelos_por_base.values())

    def _linha(r):
        base, _, pol = r.name[0], r.name[1], r.name[2]
        is_dest = (pol == politica_destaque)
        bg = fundos_destaque.get(base, '#E5E7EB') if is_dest else fundos_normais.get(base, '#FFFFFF')
        peso = '700' if is_dest else '500'
        css = (
            f"background-color: {bg}; "
            f"color: #141414; "
            f"font-weight: {peso}; "
            f"border-bottom: 1px solid #D0D7DE; "
            f"padding: 6px 10px;"
        )
        if r.name in primeiras_linhas and base != bases[0]:
            css += ' border-top: 3px solid #1F2328;'
        elif r.name in primeiras_linhas_modelo and r.name not in primeiras_linhas:
            css += ' border-top: 2px solid #8C959F;'
        return [css] * len(r)

    def _cor_lucro(v):
        cor = '#0B2545' if v >= 0 else '#5A6878'
        return f"color: {cor}; font-weight: 800;"

    def _cor_receita(_):
        return "color: #0077B6; font-weight: 700;"

    def _cor_perda(_):
        return "color: #1B2633; font-weight: 600;"

    cols_lucro = [c for c in ['Lucro Líquido', 'Lucro Treino (OOF)', 'Lucro Validação', 'Lucro Líquido Final (Unido)'] if c in df_resumo.columns]
    cols_rec   = [c for c in ['Receita VN (+2%)'] if c in df_resumo.columns]
    cols_perda = [c for c in ['Perda FP (−2%)', 'Chargeback FN (−100%)'] if c in df_resumo.columns]
    cols_usd   = cols_lucro + cols_rec + cols_perda
    cols_pct   = [c for c in ['Aceitação Geral', 'Aceitação de Pedidos', 'Aprovação', 'Aprovação (Legítimas)', 'Recall', 'Recall (Fraudes)', 'Precision'] if c in df_resumo.columns]

    styler = df_resumo.style.apply(_linha, axis=1)
    if cols_lucro:
        styler = styler.map(_cor_lucro, subset=cols_lucro)
    if cols_rec:
        styler = styler.map(_cor_receita, subset=cols_rec)
    if cols_perda:
        styler = styler.map(_cor_perda, subset=cols_perda)

    fmt_dict = {**{c: '$ {:,.0f}' for c in cols_usd}, **{c: '{:.1%}' for c in cols_pct}}
    return (
        styler
        .format(fmt_dict)
        .set_table_styles([
            {'selector': 'th.col_heading', 'props': [('background-color', '#0B2545'), ('color', '#FFFFFF'), ('font-weight', '700'), ('text-align', 'right'), ('padding', '8px 10px'), ('border', '1px solid #13579B')]},
            {'selector': 'th.index_name',  'props': [('background-color', '#0B2545'), ('color', '#FFFFFF'), ('font-weight', '700'), ('padding', '8px 10px'), ('border', '1px solid #13579B')]},
            {'selector': 'th.row_heading', 'props': [('background-color', '#13579B'), ('color', '#F5F9FD'), ('font-weight', '700'), ('padding', '6px 10px'), ('border', '1px solid #2E86C1')]},
            {'selector': 'td',             'props': [('text-align', 'right'), ('font-variant-numeric', 'tabular-nums')]},
        ])
    )


def _plot_barras_politicas(df_politicas, base, politicas, coluna, escala, fmt_rotulo, ylabel, titulo,
                           y_piso=None, figsize=(11.5, 5.8)):
    """Barras por modelo (uma por política, do tom claro ao sólido na cor do modelo) para uma métrica de uma base."""
    df_b = df_politicas.loc[base]
    modelos = list(dict.fromkeys(df_b.index.get_level_values('Modelo')))
    tons = ['clara', 'solida'] if len(politicas) == 2 else ['clara', 'media', 'solida'][:len(politicas)]
    x = np.arange(len(modelos))
    larg = 0.78 / len(politicas)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('#FCFCFD')
    valores = []
    for i, nome in enumerate(modelos):
        cores = _cor_modelo(nome, i)
        for j, pol in enumerate(politicas):
            v = df_b.loc[(nome, pol), coluna] * escala
            valores.append(v)
            xp = x[i] + (j - (len(politicas) - 1) / 2) * larg
            base_barra = y_piso if y_piso is not None else 0
            ax.bar(
                xp, v - base_barra, bottom=base_barra, width=larg * 0.88,
                color=cores[tons[j]], edgecolor='#1F2328', linewidth=0.7, zorder=3
            )
            va = 'bottom' if v >= 0 else 'top'
            ax.text(
                xp, v, f"{fmt_rotulo(v)}\n({_ROTULO_CURTO_POLITICA.get(pol, pol)})",
                ha='center', va=va, fontsize=9.8, fontweight='bold', color='#141414', zorder=5
            )

    if y_piso is None:
        lo, hi = min(0, min(valores)), max(0, max(valores))
        span = max(1.0, hi - lo)
        ax.set_ylim(lo - 0.20 * span, hi + 0.25 * span)
        ax.axhline(0, color='#1A1A1A', lw=1.0, zorder=2)
    else:
        hi = max(valores)
        ax.set_ylim(y_piso, hi + 0.25 * (hi - y_piso))
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))

    ax.set_xticks(x)
    ax.set_xticklabels(modelos, fontsize=11.5, fontweight='bold', color='#141414')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold', color='#333333', labelpad=10)
    ax.set_title(titulo, fontsize=14, fontweight='heavy', loc='left', pad=15, color='#141414')
    _estilo_limpo(ax)
    plt.tight_layout()
    plt.show()


def plot_lucro_politicas(
    df_politicas: pd.DataFrame, base: str, politicas: List[str], titulo: str = '', figsize: tuple = (11, 6),
):
    """Lucro líquido total (US$ mil) de uma base, por modelo e política."""
    _plot_barras_politicas(
        df_politicas, base, politicas, coluna='Lucro Líquido', escala=1e-3,
        fmt_rotulo=lambda v: f"US$ {v:+,.1f} mil", ylabel='Lucro líquido (US$ mil)',
        titulo=titulo or f'Lucro líquido — {base}', figsize=figsize,
    )


def plot_aceitacao_politicas(
    df_politicas: pd.DataFrame, base: str, politicas: List[str], titulo: str = '', figsize: tuple = (11, 6),
):
    """
    Taxa de aceitação geral (% de TODAS as transações aprovadas, legítimas e fraudes) de uma base,
    por modelo e política. O eixo Y começa abaixo do menor valor para as diferenças ficarem visíveis;
    o piso usado é informado no rótulo do eixo.
    """
    vals = [df_politicas.loc[(base, m, p), 'Aceitação Geral'] * 100
            for m in dict.fromkeys(df_politicas.loc[base].index.get_level_values('Modelo')) for p in politicas]
    piso = max(0.0, np.floor((min(vals) - 4) / 5) * 5)
    _plot_barras_politicas(
        df_politicas, base, politicas, coluna='Aceitação Geral', escala=100,
        fmt_rotulo=lambda v: f"{v:.1f}%", ylabel=f'Aceitação geral (%) — eixo a partir de {piso:.0f}%',
        titulo=titulo or f'Taxa de aceitação geral — {base}', y_piso=piso, figsize=figsize,
    )


def plot_tradeoff_politicas(
    df_politicas: pd.DataFrame, base: str, origem: str = 'Clássico 0.50', destino: str = 'Fixo Ótimo',
    titulo: str = '', figsize: tuple = (10, 6.5),
):
    """
    Trade-off operacional de uma base: Aprovação de legítimas (x) x Recall de fraudes (y).
    Círculo vazado = política de origem; círculo cheio = política de destino; a seta mostra o deslocamento.
    """
    df_b = df_politicas.loc[base]
    modelos = list(dict.fromkeys(df_b.index.get_level_values('Modelo')))
    fig, ax = plt.subplots(figsize=figsize)

    for i, nome in enumerate(modelos):
        cor = _cor_modelo(nome, i)['solida']
        a0, r0 = df_b.loc[(nome, origem), ['Aprovação', 'Recall']] * 100
        a1, r1 = df_b.loc[(nome, destino), ['Aprovação', 'Recall']] * 100
        ax.annotate('', xy=(a1, r1), xytext=(a0, r0),
                    arrowprops=dict(arrowstyle='->', color=cor, lw=1.8, shrinkA=8, shrinkB=8))
        ax.scatter(a0, r0, s=140, facecolor='white', edgecolor=cor, linewidth=2.2, zorder=5)
        ax.scatter(a1, r1, s=140, color=cor, edgecolor='white', linewidth=1.5, zorder=5, label=nome)
        for xa, ya, pol, peso, dy in [(a0, r0, origem, 'normal', 16), (a1, r1, destino, 'bold', -16)]:
            ax.annotate(f"{pol}\naprov. {xa:.1f}% · recall {ya:.1f}%", xy=(xa, ya), xytext=(0, dy),
                        textcoords='offset points', ha='center', va='bottom' if dy > 0 else 'top',
                        fontsize=9.5, color=cor, fontweight=peso)

    ax.margins(x=0.25, y=0.25)
    ax.set_xlabel('Aprovação de legítimas (%)  →  menos atrito com o cliente', fontsize=11)
    ax.set_ylabel('Recall de fraudes (%)  →  mais fraude barrada', fontsize=11)
    ax.set_title(titulo or f'Aprovação x captura de fraude — {base}', fontsize=14, fontweight='bold', loc='left', pad=12)
    ax.grid(ls=':', alpha=0.5)
    ax.legend(frameon=False, fontsize=10, loc='lower left')
    _limpar_bordas(ax)
    plt.tight_layout()
    plt.show()


def tabela_risco_inerente(particoes: Dict[str, pd.DataFrame], col_target: str = 'isFraud') -> pd.DataFrame:
    """
    Calcula a DRE (Demonstrativo de Resultado) teórica assumindo que a empresa
    aprovasse 100% ou bloqueasse 100% das transações, sem usar nenhum modelo.
    """
    linhas = []
    for nome, df in particoes.items():
        # Valores reais em US$ (reverte log1p se necessário)
        amt = obter_amt_real(df)
        y = df[col_target].values

        vol_legit = float(amt[y == 0].sum())
        vol_fraud = float(amt[y == 1].sum())

        linhas.append({
            'Base': nome,
            'Transações': len(y),
            'Taxa Fraude': y.mean(),
            'Volume Legítimo (US$)': vol_legit,
            'Volume Fraude (US$)': vol_fraud,
            'Receita Potencial (+2%)': vol_legit * 0.02,
            'Prejuízo Inerente (-100%)': vol_fraud,
            'LUCRO: Aprovar 100%': (vol_legit * 0.02) - vol_fraud,
            'LUCRO: Bloquear 100%': -(vol_legit * 0.02),
        })

    df_res = pd.DataFrame(linhas)

    # Linha de Total
    total = df_res.sum(numeric_only=True)
    total['Base'] = 'TOTAL'
    total['Taxa Fraude'] = (df_res['Taxa Fraude'] * df_res['Transações']).sum() / df_res['Transações'].sum()
    df_res = pd.concat([df_res, total.to_frame().T], ignore_index=True)

    return df_res


def plot_dilema_financeiro(df_tabela: pd.DataFrame, figsize: tuple = (12, 4.5)):
    """
    Plota as barras contrapostas do cenário base com rótulos de dados diretos,
    com limites simétricos forçados para evitar sobreposição de texto.
    """
    # Isola a linha de TOTAL da tabela
    tot = df_tabela[df_tabela['Base'] == 'TOTAL'].iloc[0]
    
    receita = tot['Receita Potencial (+2%)'] / 1e6
    prejuizo = tot['Prejuízo Inerente (-100%)'] / 1e6
    lucro_aprov = tot['LUCRO: Aprovar 100%'] / 1e6
    lucro_bloq = tot['LUCRO: Bloquear 100%'] / 1e6
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = [1.2, 0] # Espaçamento vertical entre cenários
    labels = [
        f"CENÁRIO 1: Aprovar 100%\nLucro Líquido: -US$ {abs(lucro_aprov):.2f} Mi", 
        f"CENÁRIO 2: Bloquear 100%\nLucro Líquido: -US$ {abs(lucro_bloq):.2f} Mi"
    ]
    
    altura_barra = 0.45
    ax.barh(y_pos[0], receita, color='#0096C7', height=altura_barra, label='Receita Potencial (+2% Legítimas)')
    ax.barh(y_pos[0], -prejuizo, color='#0B2545', height=altura_barra, label='Chargeback Fraude (-100%)')
    ax.barh(y_pos[1], -receita, color='#5DADE2', height=altura_barra, label='Asfixia Comercial (Falsos Positivos)')
    
    # Linha central âncora
    ax.axvline(0, color='#1A1A1A', lw=2)
    
    # ==========================================================================
    # CORREÇÃO DE ESPAÇAMENTO: Limites simétricos dinâmicos
    # ==========================================================================
    max_val = max(receita, prejuizo)
    ax.set_xlim(-max_val * 1.4, max_val * 1.4) # Adiciona 40% de margem nas laterais
    offset = max_val * 0.03 # Afastamento relativo para os rótulos de dados
    
    # Rótulos de dados cravados nas pontas das barras
    ax.text(receita + offset, y_pos[0], f"+US$ {receita:.2f}M", va='center', ha='left', color='#0096C7', fontweight='bold', fontsize=11)
    ax.text(-prejuizo - offset, y_pos[0], f"-US$ {prejuizo:.2f}M", va='center', ha='right', color='#0B2545', fontweight='bold', fontsize=11)
    ax.text(-receita - offset, y_pos[1], f"-US$ {receita:.2f}M", va='center', ha='right', color='#2E86C1', fontweight='bold', fontsize=11)
    
    # Formatação dos eixos e títulos
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontweight='bold', fontsize=11)
    ax.set_xticks([])  # Removemos os números do eixo X
    
    # Afasta os labels do eixo Y para não colidirem com os números vermelhos
    ax.tick_params(axis='y', pad=15)
    
    ax.set_title("O Problema de Negócio (Risco Inerente Total)", loc='left', fontweight='heavy', fontsize=14, pad=20)
    
    # Legenda alinhada e discreta
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False, fontsize=10)
    
    # Remoção de TODAS as bordas (Spines)
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    plt.tight_layout()
    plt.show()

def avaliar_politicas_por_bloco(y_true, amt, blocos, politicas: Dict[str, Any], rotulo_bloco: str = 'Mês') -> pd.DataFrame:
    """
    Score financeiro de cada política de bloqueio (dict nome -> máscara booleana de bloqueio)
    em cada bloco temporal e no total do período.
    """
    y_true = np.asarray(y_true)
    amt = np.asarray(amt, dtype=float)
    blocos = np.asarray(blocos)
    linhas = []
    for b in list(np.unique(blocos)) + ['Total']:
        msk = np.ones(len(y_true), dtype=bool) if b == 'Total' else (blocos == b)
        for nome, bloqueio in politicas.items():
            r = calcular_score_financeiro(y_true[msk], np.asarray(bloqueio, dtype=bool)[msk], amt[msk])
            linhas.append({
                'Bloco': 'Total' if b == 'Total' else f'{rotulo_bloco} {b}',
                'Politica': nome,
                'Lucro': r['Resultado_Liquido'],
                'Receita_VN': r['Receita_VN'],
                'Penalidade_FP': r['Penalidade_FP'],
                'Prejuizo_FN': r['Prejuizo_FN'],
                'Aprovacao': r['Approval_Rate'],
                'Recall': r['Recall'],
                'Precision': r['Precision'],
            })
    return pd.DataFrame(linhas)


def plot_estabilidade_temporal(df_blocos: pd.DataFrame, titulo: str = '', figsize: tuple = (18, 5.4)):
    """
    Painel triplo de estabilidade mensal (exclui a linha Total):
      (1) Resultado Líquido das Políticas Ativas (US$ Mil) com rótulos diretos (anotando o prejuízo Sem Modelo);
      (2) Recall de Fraudes Bloqueadas (%) por bloco com rótulos diretos;
      (3) Aprovação de Legítimas (%) por bloco com escala ajustada e rótulos diretos.
    """
    df_plot = df_blocos[df_blocos['Bloco'] != 'Total'].copy()
    blocos = list(dict.fromkeys(df_plot['Bloco']))
    politicas_todas = list(dict.fromkeys(df_plot['Politica']))

    # Separa "Sem Modelo" no gráfico de Lucro se o prejuízo for muito discrepante, para não esmagar as barras dos modelos
    pols_modelos = [p for p in politicas_todas if p != 'Sem Modelo']
    usar_foco_modelos = len(pols_modelos) >= 2 and ('Sem Modelo' in politicas_todas)

    mapa_cores = {
        'Sem Modelo': '#97A3B0',
        'Clássico 0.50': '#5DADE2',
        'Fixo Ótimo': '#0096C7',
        'Dinâmico (Optuna)': '#0B2545',
    }
    for idx_c, p in enumerate(politicas_todas):
        if p not in mapa_cores:
            mapa_cores[p] = _PALETA_POLITICAS[idx_c % len(_PALETA_POLITICAS)]

    x = np.arange(len(blocos))
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # --- Painel 1: Resultado Líquido por Mês (US$ Mil) ---
    pols_p1 = pols_modelos if usar_foco_modelos else politicas_todas
    larg_p1 = 0.76 / max(1, len(pols_p1))
    vals_p1 = []
    for i, pol in enumerate(pols_p1):
        d = df_plot[df_plot['Politica'] == pol].set_index('Bloco').loc[blocos]
        y_k = d['Lucro'].values / 1e3
        vals_p1.extend(y_k)
        pos = x + (i - (len(pols_p1) - 1) / 2) * larg_p1
        bars = axes[0].bar(pos, y_k, width=larg_p1 * 0.92, color=mapa_cores[pol], label=pol)
        for bar, val in zip(bars, y_k):
            va = 'bottom' if val >= 0 else 'top'
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                val + (0.8 if val >= 0 else -0.8),
                f"${val:+,.1f}k",
                ha='center', va=va, fontsize=8.6, fontweight='bold', color='#1A1A1A'
            )

    axes[0].axhline(0, color='#1A1A1A', lw=1.0)
    if vals_p1:
        ymin1, ymax1 = min(0.0, min(vals_p1)), max(0.0, max(vals_p1))
        span1 = max(5.0, ymax1 - ymin1)
        axes[0].set_ylim(ymin1 - 0.25 * span1, ymax1 + 0.30 * span1)

    if usar_foco_modelos:
        d_sem = df_plot[df_plot['Politica'] == 'Sem Modelo'].set_index('Bloco').loc[blocos]
        resumo_sem = " | ".join([f"{b}: -US$ {abs(v)/1e3:,.0f}k" for b, v in zip(blocos, d_sem['Lucro'].values)])
        axes[0].text(
            0.02, 0.96, f"Ref. Sem Modelo (Aprovar 100%): {resumo_sem}",
            transform=axes[0].transAxes, ha='left', va='top', fontsize=8.3, fontweight='bold', color='#555555',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5F5', edgecolor='#BDBDBD', alpha=0.92)
        )

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(blocos, fontweight='bold')
    axes[0].set_title('1. Resultado Líquido por Mês (Modelos)', fontsize=11.5, fontweight='bold', loc='left')
    axes[0].set_ylabel('US$ Mil')
    axes[0].grid(axis='y', linestyle=':', alpha=0.5)
    axes[0].legend(frameon=False, fontsize=8.8, loc='lower right')
    _limpar_bordas(axes[0])

    # --- Painel 2: Recall de Fraudes Bloqueadas (%) ---
    pols_p2 = pols_modelos if usar_foco_modelos else politicas_todas
    larg_p2 = 0.76 / max(1, len(pols_p2))
    vals_p2 = []
    for i, pol in enumerate(pols_p2):
        d = df_plot[df_plot['Politica'] == pol].set_index('Bloco').loc[blocos]
        y_rec = d['Recall'].values * 100
        vals_p2.extend(y_rec)
        pos = x + (i - (len(pols_p2) - 1) / 2) * larg_p2
        bars = axes[1].bar(pos, y_rec, width=larg_p2 * 0.92, color=mapa_cores[pol], label=pol)
        for bar, val in zip(bars, y_rec):
            axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.8, f"{val:.1f}%",
                         ha='center', va='bottom', fontsize=8.6, fontweight='bold', color='#1A1A1A')

    if vals_p2:
        min_r = max(0.0, min(vals_p2) - 12.0)
        axes[1].set_ylim(min_r, min(105.0, max(vals_p2) + 10.0))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(blocos, fontweight='bold')
    axes[1].set_title('2. Recall (Fraudes Bloqueadas %)', fontsize=11.5, fontweight='bold', loc='left')
    axes[1].set_ylabel('Recall (%)')
    axes[1].grid(axis='y', linestyle=':', alpha=0.5)
    axes[1].legend(frameon=False, fontsize=8.8, loc='lower right')
    _limpar_bordas(axes[1])

    # --- Painel 3: Taxa de Aprovação de Legítimas (%) ---
    larg_p3 = 0.78 / max(1, len(politicas_todas))
    vals_p3 = []
    for i, pol in enumerate(politicas_todas):
        d = df_plot[df_plot['Politica'] == pol].set_index('Bloco').loc[blocos]
        y_apr = d['Aprovacao'].values * 100
        vals_p3.extend(y_apr)
        pos = x + (i - (len(politicas_todas) - 1) / 2) * larg_p3
        bars = axes[2].bar(pos, y_apr, width=larg_p3 * 0.92, color=mapa_cores[pol], label=pol)
        for bar, val in zip(bars, y_apr):
            axes[2].text(bar.get_x() + bar.get_width() / 2, val + 0.5, f"{val:.1f}%",
                         ha='center', va='bottom', fontsize=8.3, fontweight='bold', color='#1A1A1A')

    if vals_p3:
        min_a = max(0.0, min(vals_p3) - 6.0)
        axes[2].set_ylim(min_a, 104.5)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(blocos, fontweight='bold')
    axes[2].set_title('3. Aprovação de Legítimas (%)', fontsize=11.5, fontweight='bold', loc='left')
    axes[2].set_ylabel('Aprovação (%)')
    axes[2].grid(axis='y', linestyle=':', alpha=0.5)
    axes[2].legend(frameon=False, fontsize=8.5, loc='lower right')
    _limpar_bordas(axes[2])

    if titulo:
        fig.suptitle(titulo, fontsize=13, fontweight='bold', x=0.01, ha='left', y=1.02)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------
# THRESHOLD DINÂMICO POR TICKET (VP=0, VN=+0.02, FP=-0.02, FN=-1.0)
# ------------------------------------------------------------------------------
def calcular_matriz_lucro(y_true, prob, threshold_array, amt):
    """Função de lucro estrita (Matriz: VP=0, VN=0.02, FP=-0.02, FN=-1.0) para threshold fixo ou vetorizado."""
    y_true = np.asarray(y_true)
    amt = np.asarray(amt, dtype=float)
    bloqueado = np.asarray(prob) >= threshold_array
    aprovado = ~bloqueado

    vn = aprovado  & (y_true == 0)
    fp = bloqueado & (y_true == 0)
    fn = aprovado  & (y_true == 1)

    receita_vn  = (0.02 * amt[vn]).sum()
    penal_fp    = (0.02 * amt[fp]).sum()
    prejuizo_fn = (1.00 * amt[fn]).sum()

    lucro_liquido = receita_vn - penal_fp - prejuizo_fn

    return {
        'Lucro': lucro_liquido,
        'Aprovacao': vn.sum() / max(1, (y_true == 0).sum()),
        'Recall': (bloqueado & (y_true == 1)).sum() / max(1, (y_true == 1).sum())
    }


def threshold_dinamico(amt, t_min, t_max, k):
    """Curva de threshold por ticket: decai de t_max (micro-compras) até t_min (alto ticket)."""
    return t_min + (t_max - t_min) * np.exp(-k * np.asarray(amt, dtype=float))


def objective_threshold_dinamico(trial, y_true, probs, amt):
    """Função objetivo Optuna: maximiza o lucro do threshold dinâmico no OOF calibrado."""
    t_max = trial.suggest_float('t_max', 0.05, 0.35)           # Teto para tickets muito baixos
    t_min = trial.suggest_float('t_min', 0.005, t_max - 0.01)  # Piso para tickets altos (sempre menor que t_max)
    k     = trial.suggest_float('k', 0.001, 0.05, log=True)    # Taxa de queda

    th = threshold_dinamico(amt, t_min, t_max, k)
    return calcular_matriz_lucro(y_true, probs, th, amt)['Lucro']


def plot_curva_threshold_dinamico(
    th_dinamico: Dict[str, Dict[str, float]],
    threshold_otimo: Dict[str, float],
    nome_modelo: Optional[str] = None,
    amt_max: float = 1500,
    titulo: str = '',
    figsize: Optional[tuple] = None,
):
    """
    Gráfico limpo e minimalista da curva de threshold dinâmico por valor da transação.
    Se `nome_modelo` for informado, plota apenas aquele modelo; caso contrário, plota lado a lado.
    """
    nomes = [nome_modelo] if nome_modelo is not None else list(th_dinamico.keys())
    n_mod = len(nomes)
    figsize = figsize or ((11, 4.6) if n_mod == 1 else (7.5 * n_mod, 4.6))
    eixo_amt = np.linspace(0, amt_max, 400)

    fig, axes = plt.subplots(1, n_mod, figsize=figsize, sharey=True)
    if n_mod == 1:
        axes = [axes]

    max_y_value = max(max(d['t_max'], threshold_otimo.get(n, 0.0)) for n, d in th_dinamico.items())
    y_top = max_y_value * 1.18

    for i, (ax, nome) in enumerate(zip(axes, nomes)):
        d = th_dinamico[nome]
        th_fixo = float(threshold_otimo[nome])
        cor = _cor_modelo(nome, i)['solida']
        curva = threshold_dinamico(eixo_amt, d['t_min'], d['t_max'], d['k'])
        th_fim = float(curva[-1])

        ax.set_xlim(0, amt_max)
        ax.set_ylim(0, y_top)

        # Sombreado leve na zona de aprovação (abaixo da curva)
        ax.fill_between(eixo_amt, 0, curva, color=cor, alpha=0.05, lw=0)

        # Curva dinâmica + linhas de referência (informações na legenda, sem caixas no meio do gráfico)
        ax.plot(eixo_amt, curva, color=cor, lw=2.8,
                label=f'Curva Dinâmica ({d["t_max"]:.2%} → {d["t_min"]:.2%})')
        ax.axhline(th_fixo, color='#555555', ls='--', lw=1.5,
                   label=f'Fixo Ótimo ({th_fixo:.2%})')
        ax.axhline(d['t_min'], color='#999999', ls=':', lw=1.2,
                   label=f'Assíntota Alto Ticket ({d["t_min"]:.2%})')

        # Apenas marcadores discretos nas extremidades
        ax.scatter([0, amt_max], [d['t_max'], th_fim], color=cor, s=55, edgecolor='white', linewidth=1.5, zorder=5)
        ax.annotate(f"{d['t_max']:.2%}", xy=(0, d['t_max']), xytext=(8, 4), textcoords='offset points',
                    fontsize=9.5, fontweight='bold', color=cor, ha='left', va='bottom')
        ax.annotate(f"{th_fim:.2%}", xy=(amt_max, th_fim), xytext=(-8, 6), textcoords='offset points',
                    fontsize=9.5, fontweight='bold', color=cor, ha='right', va='bottom')

        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('US$ {x:,.0f}'))
        ax.set_xlabel('Valor da transação (US$)', fontsize=11, color='#333333', labelpad=8)
        if i == 0:
            ax.set_ylabel('Threshold de bloqueio exigido', fontsize=11, color='#333333', labelpad=8)
        else:
            ax.set_ylabel('')
            ax.tick_params(labelleft=True)

        ax.set_title(
            titulo if (titulo and n_mod == 1) else f'Threshold Dinâmico vs. Fixo — {nome}',
            fontsize=13, fontweight='heavy', loc='left', pad=12, color='#1A1A1A',
        )
        ax.legend(frameon=False, fontsize=9.5, loc='upper right')
        _estilo_limpo(ax)

    if titulo and n_mod > 1:
        fig.suptitle(titulo, fontsize=13.5, fontweight='heavy', x=0.01, ha='left', y=1.02, color='#1A1A1A')
    plt.tight_layout()
    plt.show()


def tabela_consolidada_treino_validacao(
    y_train, amt_train, politicas_treino: Dict[str, Dict[str, np.ndarray]],
    y_val, amt_val, politicas_val: Dict[str, Dict[str, np.ndarray]],
) -> pd.DataFrame:
    """
    Constrói a tabela final da Seção 7.4 unindo Treino (OOF) + Validação para LGBM_Max e LGBM_Otimizado
    nas 4 políticas (Sem Algoritmo / Risco, Clássico 0.50, Fixo Ótimo e Dinâmico Optuna), exibindo:
      - Lucro no Treino (OOF), Lucro na Validação e Lucro Líquido Final (Unido);
      - Receita VN (+2%), Perda FP (−2%) e Chargeback FN (−100%) acumulados;
      - Aceitação de Pedidos (% geral de pedidos aceitos), Aprovação (Legítimas), Recall e Precision.
    """
    y_tr_arr, amt_tr_arr = np.asarray(y_train), np.asarray(amt_train, dtype=float)
    y_va_arr, amt_va_arr = np.asarray(y_val),   np.asarray(amt_val, dtype=float)
    y_unido   = np.concatenate([y_tr_arr, y_va_arr])
    amt_unido = np.concatenate([amt_tr_arr, amt_va_arr])

    nomes_exibicao = {
        'Sem Modelo': 'Sem Algoritmo / Risco',
        'Clássico 0.50': 'Corte 0.50',
        'Fixo Ótimo': 'Fixo Ótimo',
        'Dinâmico (Optuna)': 'Dinâmico (Optuna)',
    }

    linhas = []
    for nome in politicas_treino.keys():
        for pol_chave in politicas_treino[nome].keys():
            bloq_tr = np.asarray(politicas_treino[nome][pol_chave], dtype=bool)
            bloq_va = np.asarray(politicas_val[nome][pol_chave], dtype=bool)
            bloq_un = np.concatenate([bloq_tr, bloq_va])

            r_tr = calcular_score_financeiro(y_tr_arr, bloq_tr, amt_tr_arr)
            r_va = calcular_score_financeiro(y_va_arr, bloq_va, amt_va_arr)
            r_un = calcular_score_financeiro(y_unido,  bloq_un, amt_unido)

            linhas.append({
                'Base': 'Treino + Validação (Unido)',
                'Modelo': nome,
                'Política': nomes_exibicao.get(pol_chave, pol_chave),
                'Lucro Treino (OOF)': r_tr['Resultado_Liquido'],
                'Lucro Validação': r_va['Resultado_Liquido'],
                'Lucro Líquido Final (Unido)': r_un['Resultado_Liquido'],
                'Receita VN (+2%)': r_un['Receita_VN'],
                'Perda FP (−2%)': r_un['Penalidade_FP'],
                'Chargeback FN (−100%)': r_un['Prejuizo_FN'],
                'Aceitação de Pedidos': 1.0 - (r_un['Qtd_VP'] + r_un['Qtd_FP']) / max(1, len(bloq_un)),
                'Aprovação (Legítimas)': r_un['Approval_Rate'],
                'Recall (Fraudes)': r_un['Recall'],
                'Precision': r_un['Precision'],
            })

    return pd.DataFrame(linhas).set_index(['Base', 'Modelo', 'Política'])


# ------------------------------------------------------------------------------
# MODELO FINAL (SEÇÃO 8): ARTEFATO, REGRA DINÂMICA, AVALIAÇÃO E EXPORTAÇÃO
# ------------------------------------------------------------------------------
def montar_artefato_final(
    nome_modelo: str, modelo, calibrador, features: List[str],
    params_dinamico: Dict[str, float], threshold_fixo_ref: Optional[float] = None,
    n_amostras_treino: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Empacota tudo o que a inferência precisa em um único objeto serializável:
    estimador (treinado em 100% do Treino), calibrador isotônico, lista de features e parâmetros da regra dinâmica.
    """
    return {
        'nome_modelo': nome_modelo,
        'modelo': modelo,
        'calibrador': calibrador,
        'features': list(features),
        'treinado_100pct_treino': True,
        'n_amostras_treino': None if n_amostras_treino is None else int(n_amostras_treino),
        'regra_decisao': {
            'tipo': 'threshold_dinamico',
            'formula': 'bloquear se prob_calibrada >= t_min + (t_max - t_min) * exp(-k * TransactionAmt_USD)',
            't_min': float(params_dinamico['t_min']),
            't_max': float(params_dinamico['t_max']),
            'k': float(params_dinamico['k']),
            'lucro_oof_optuna': None if 'lucro_oof' not in params_dinamico else float(params_dinamico['lucro_oof']),
            'threshold_fixo_referencia': None if threshold_fixo_ref is None else float(threshold_fixo_ref),
        },
    }


def aplicar_regra_dinamica(artefato: Dict[str, Any], X, amt_real) -> Dict[str, np.ndarray]:
    """
    Inferência com o artefato final: probabilidade bruta -> calibrada -> corte por valor -> decisão.
    `amt_real` é o valor da transação em US$ reais (não em log).
    """
    X_ = X[artefato['features']] if hasattr(X, 'columns') else X
    regra = artefato['regra_decisao']
    prob_raw = artefato['modelo'].predict_proba(X_)[:, 1]
    prob_cal = artefato['calibrador'].transform(prob_raw)
    th = threshold_dinamico(amt_real, regra['t_min'], regra['t_max'], regra['k'])
    return {'prob_raw': prob_raw, 'prob_cal': prob_cal, 'threshold': th, 'bloquear': prob_cal >= th}


# (rótulo, chave, tipo, mostra gap) — o gap só faz sentido para métricas que não dependem do tamanho da base
_LAYOUT_AVALIACAO_FINAL = {
    'METRICAS INDEPENDENTES DO THRESHOLD (Probabilidade bruta)': [
        ('ROC-AUC', 'ROC-AUC', 'prob', True),
        ('GINI', 'GINI', 'prob', True),
        ('PR-AUC', 'PR-AUC', 'prob', True),
        ('KS', 'KS', 'prob', True),
    ],
    'METRICAS DA REGRA DE DECISAO (Threshold dinâmico)': [
        ('Precision', 'Precision', 'pct', True),
        ('Recall', 'Recall', 'pct', True),
        ('F1-Score', 'F1-Score', 'prob', True),
        ('Aprovação Legítimas', 'Aprovação', 'pct', True),
        ('Aceitação Geral', 'Aceitação Geral', 'pct', True),
        ('Fraudes Barradas (VP)', 'Qtd_VP', 'int', False),
        ('Legítimas Bloq. (FP)', 'Qtd_FP', 'int', False),
        ('Fraudes Aprovadas (FN)', 'Qtd_FN', 'int', False),
    ],
    'RESULTADO FINANCEIRO (US$)': [
        ('Receita VN (+2%)', 'Receita VN', 'usd', False),
        ('Perda FP (-2%)', 'Perda FP', 'usd', False),
        ('Chargeback FN (-100%)', 'Chargeback FN', 'usd', False),
        ('Lucro Líquido', 'Lucro Líquido', 'usd', False),
        ('Lucro Sem Modelo (ref.)', 'Lucro Sem Modelo', 'usd', False),
        ('Lucro por Transação', 'Lucro por Transação', 'usd3', True),
        ('Margem s/ Volume', 'Margem s/ Volume', 'pct', True),
    ],
}


def _metricas_particao_unica(y, prob_raw, bloquear, amt) -> Dict[str, float]:
    """Calcula métricas estatísticas, operacionais e financeiras em um único vetor/partição."""
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    y = np.asarray(y).astype(int)
    prob_raw = np.asarray(prob_raw, dtype=float)
    bloquear = np.asarray(bloquear, dtype=bool)
    amt = np.asarray(amt, dtype=float)

    roc = roc_auc_score(y, prob_raw) if len(np.unique(y)) > 1 else 0.0
    pr  = average_precision_score(y, prob_raw) if len(np.unique(y)) > 1 else 0.0
    ks  = ks_2samp(prob_raw[y == 1], prob_raw[y == 0]).statistic if (y == 1).any() and (y == 0).any() else 0.0
    r = calcular_score_financeiro(y, bloquear, amt)
    r_sem = calcular_score_financeiro(y, np.zeros(len(y), dtype=bool), amt)
    return {
        'Transações': int(len(y)),
        'Taxa Fraude': float(y.mean()),
        'ROC-AUC': float(roc),
        'GINI': float(2 * roc - 1),
        'PR-AUC': float(pr),
        'KS': float(ks),
        'Precision': float(r['Precision']),
        'Recall': float(r['Recall']),
        'F1-Score': float(f1_score(y, bloquear.astype(int), zero_division=0)),
        'Aprovação': float(r['Approval_Rate']),
        'Aceitação Geral': float(1 - (r['Qtd_VP'] + r['Qtd_FP']) / max(1, len(y))),
        'Qtd_VP': int(r['Qtd_VP']), 'Qtd_FP': int(r['Qtd_FP']), 'Qtd_FN': int(r['Qtd_FN']), 'Qtd_VN': int(r['Qtd_VN']),
        'Receita VN': float(r['Receita_VN']),
        'Perda FP': float(r['Penalidade_FP']),
        'Chargeback FN': float(r['Prejuizo_FN']),
        'Lucro Líquido': float(r['Resultado_Liquido']),
        'Lucro Sem Modelo': float(r_sem['Resultado_Liquido']),
        'Lucro por Transação': float(r['Resultado_Liquido'] / max(1, len(y))),
        'Margem s/ Volume': float(r['Resultado_Liquido'] / max(1.0, amt.sum())),
    }


def _metricas_modelo_final(y, prob_raw, bloquear, amt, groups=None) -> Dict[str, float]:
    """
    Métricas de ranqueamento, da regra de decisão e financeiras de uma base.
    Se `groups` for informado (ex.: blocos do CV no Treino), as métricas relativas/taxas
    (ROC-AUC, GINI, PR-AUC, KS, Precision, Recall, F1, Aprovação, Aceitação, Lucro/Tx e Margem)
    são calculadas como a MÉDIA DOS TESTES DO CV, mantendo os volumes totais somados no OOF.
    """
    base_global = _metricas_particao_unica(y, prob_raw, bloquear, amt)
    if groups is None:
        return base_global

    y_arr = np.asarray(y).astype(int)
    p_arr = np.asarray(prob_raw, dtype=float)
    b_arr = np.asarray(bloquear, dtype=bool)
    a_arr = np.asarray(amt, dtype=float)
    g_arr = np.asarray(groups)

    gkf = GroupKFold(n_splits=min(4, len(np.unique(g_arr))))
    metricas_folds = []
    for _, va_idx in gkf.split(p_arr, y_arr, groups=g_arr):
        if len(np.unique(y_arr[va_idx])) > 1:
            metricas_folds.append(_metricas_particao_unica(y_arr[va_idx], p_arr[va_idx], b_arr[va_idx], a_arr[va_idx]))

    if not metricas_folds:
        return base_global

    chaves_media_cv = [
        'ROC-AUC', 'GINI', 'PR-AUC', 'KS',
        'Precision', 'Recall', 'F1-Score', 'Aprovação', 'Aceitação Geral',
        'Lucro por Transação', 'Margem s/ Volume',
    ]
    for k in chaves_media_cv:
        base_global[k] = float(np.mean([m[k] for m in metricas_folds]))
    return base_global


def avaliar_modelo_final(artefato: Dict[str, Any], bases: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Avaliação final do modelo campeão com a regra dinâmica (sem tocar no OOT):
      - `Treino (Teste CV)`: média dos folds de teste do CV no Treino (e totais OOF);
      - `Validação`: base isolada de validação (`df_validacao`);
      - `Gap (Val - CV)`: diferença entre Validação e Treino (Teste CV).
    Retorna um DataFrame com as colunas das bases e a coluna `Gap (Val - CV)`.
    """
    regra = artefato['regra_decisao']
    metricas = {}
    for nome_base, b in bases.items():
        th = threshold_dinamico(b['amt'], regra['t_min'], regra['t_max'], regra['k'])
        bloquear = np.asarray(b['prob_cal']) >= th
        metricas[nome_base] = _metricas_modelo_final(
            b['y'], b['prob_raw'], bloquear, b['amt'], groups=b.get('groups')
        )
    df = pd.DataFrame(metricas)

    nomes = list(bases)
    ref, alvo = (nomes[-2], nomes[-1]) if len(nomes) >= 2 else (nomes[0], nomes[0])
    col_gap = 'Gap (Val - CV)' if len(nomes) == 2 else f'Gap ({alvo} - {ref[:3]})'
    df[col_gap] = df[alvo] - df[ref]

    chaves_sem_gap = {
        chave for linhas in _LAYOUT_AVALIACAO_FINAL.values()
        for _, chave, _, tem_gap in linhas if not tem_gap
    } | {'Transações'}
    for ch in chaves_sem_gap:
        if ch in df.index:
            df.loc[ch, col_gap] = np.nan

    larg = 26 + 20 * (len(nomes) + 1)

    def _fmt(v, tipo):
        return {'prob': f"{v:.4f}", 'pct': f"{v:.2%}", 'int': f"{v:,.0f}",
                'usd': f"$ {v:,.0f}", 'usd3': f"$ {v:,.3f}"}[tipo]

    def _fmt_gap(v, tipo):
        if pd.isna(v):
            return '-'
        return {'prob': f"{v:+.4f}", 'pct': f"{v * 100:+.2f} p.p.", 'usd3': f"$ {v:+,.3f}"}.get(tipo, '-')

    print('=' * larg)
    print(f" MODELO FINAL: {artefato['nome_modelo'].upper()} | {len(artefato['features'])} features | Threshold dinâmico")
    print(f" Regra: bloquear se prob. calibrada >= {regra['t_min']:.4f} + ({regra['t_max']:.4f} - {regra['t_min']:.4f}) · exp(-{regra['k']:.5f} · Valor US$)")
    print('-' * larg)
    print(" Taxa Base de Fraude (Target=1): " + " | ".join(
        f"{n} = {df.loc['Taxa Fraude', n]:.2%} ({int(df.loc['Transações', n]):,} tx)" for n in nomes))
    print('-' * larg)

    cab = f" {'Metrica':<24}" + "".join(f" | {n:>18}" for n in nomes) + f" | {col_gap:>16}"
    for bloco, linhas in _LAYOUT_AVALIACAO_FINAL.items():
        print(f"\n [ {bloco} ]")
        print(cab)
        print('-' * len(cab))
        for rotulo, chave, tipo, gap in linhas:
            vals = "".join(f" | {_fmt(df.loc[chave, n], tipo):>18}" for n in nomes)
            g = _fmt_gap(df.loc[chave, col_gap], tipo) if gap else '-'
            print(f" {rotulo:<24}{vals} | {g:>16}")
    print('=' * larg + '\n')
    return df
def plot_matriz_confusao_final(
    artefato: Dict[str, Any],
    bases: Dict[str, Dict[str, Any]],
    figsize: Tuple[float, float] = (7, 5),
) -> pd.DataFrame:
    """
    Gera uma Matriz de Confusão minimalista para o conjunto Treino + Validação.
    Foca nos valores absolutos e percentuais relevantes, acompanhada de 
    uma tabela de resumo direta.
    """
    regra = artefato['regra_decisao']
    y_particoes, pred_particoes = {}, {}

    # Extrai e calcula predições
    for nome, b in bases.items():
        th = threshold_dinamico(b['amt'], regra['t_min'], regra['t_max'], regra['k'])
        pred_particoes[nome] = (np.asarray(b['prob_cal']) >= th).astype(int)
        y_particoes[nome] = np.asarray(b['y']).astype(int)

    # Consolida Treino + Validação
    y_total = np.concatenate(list(y_particoes.values()))
    pred_total = np.concatenate(list(pred_particoes.values()))

    # Calcula a Matriz
    cm = confusion_matrix(y_total, pred_total, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    n_legit, n_fraude = tn + fp, fn + tp
    n_total = n_legit + n_fraude

    # Estruturação para o Heatmap (Percentual da Classe Real)
    cm_pct = np.array([
        [tn / n_legit, fp / n_legit],
        [fn / n_fraude, tp / n_fraude]
    ])

    # Plot Minimalista
    fig, ax = plt.subplots(figsize=figsize)
    
    # Heatmap corporativo (Vermelho para Erros, Cinza Escuro para Acertos na Fraude)
    sns.heatmap(
        cm_pct, annot=False, cmap='Blues', vmin=0, vmax=1, 
        cbar=False, linewidths=2, linecolor='white', ax=ax
    )

    # Rótulos customizados nos 4 quadrantes
    labels = [
        ("TN: Legítima Aprovada", tn, cm_pct[0, 0]),
        ("FP: Falso Alarme", fp, cm_pct[0, 1]),
        ("FN: Fraude Passou", fn, cm_pct[1, 0]),
        ("VP: Fraude Bloqueada", tp, cm_pct[1, 1])
    ]
    
    for i, (nome, abs_val, pct_val) in enumerate(labels):
        row, col = divmod(i, 2)
        cor_txt = "white" if pct_val > 0.5 else "#333333"
        ax.text(col + 0.5, row + 0.45, f"{pct_val:.1%}", ha="center", va="center", color=cor_txt, fontsize=16, fontweight='bold')
        ax.text(col + 0.5, row + 0.65, f"{nome}\nn = {abs_val:,}", ha="center", va="center", color=cor_txt, fontsize=10)

    # Formatação limpa dos eixos
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(['Aprovada (0)', 'Bloqueada (1)'], fontsize=11, fontweight='bold')
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(['Legítima (0)', 'Fraude (1)'], fontsize=11, fontweight='bold', va='center')
    
    ax.set_xlabel('Decisão do Modelo', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Classe Real', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f"Matriz de Confusão: \nTreino + Validação", 
                 loc='left', fontsize=13, fontweight='heavy', pad=15)
    
    plt.tight_layout()
    plt.show()

    # Tabela de Resumo Flat e Direta
    df_resumo = pd.DataFrame([
        {'Partição': 'Consolidado', 'Classe Real': 'Legítima (0)', 'Decisão': 'Aprovada (TN)', 'Transações': tn, '% da Classe': f"{tn/n_legit:.2%}"},
        {'Partição': 'Consolidado', 'Classe Real': 'Legítima (0)', 'Decisão': 'Bloqueada (FP)', 'Transações': fp, '% da Classe': f"{fp/n_legit:.2%}"},
        {'Partição': 'Consolidado', 'Classe Real': 'Fraude (1)',   'Decisão': 'Aprovada (FN)', 'Transações': fn, '% da Classe': f"{fn/n_fraude:.2%}"},
        {'Partição': 'Consolidado', 'Classe Real': 'Fraude (1)',   'Decisão': 'Bloqueada (VP)', 'Transações': tp, '% da Classe': f"{tp/n_fraude:.2%}"},
    ])
    
    return df_resumo.set_index(['Partição', 'Classe Real', 'Decisão'])


def _serializar_params(modelo) -> Dict[str, Any]:
    """Hiperparâmetros do estimador em tipos nativos (JSON-safe)."""
    saida = {}
    for k, v in modelo.get_params(deep=False).items():
        if isinstance(v, np.integer):
            saida[k] = int(v)
        elif isinstance(v, np.floating):
            saida[k] = float(v)
        elif isinstance(v, (int, float, str, bool, list, dict)) or v is None:
            saida[k] = v
        else:
            saida[k] = str(v)
    return saida


def exportar_modelo_final(
    artefato: Dict[str, Any], df_metricas: pd.DataFrame, diretorio_destino, nome_arquivo: str = 'modelo_fraude_final',
) -> Dict[str, Path]:
    """
    Exporta o modelo final (treinado em 100% da base de Treino) e seus relatórios completos:
      - <nome>.joblib          : artefato completo (modelo 100% treino + calibrador + features + regra dinâmica);
      - <nome>_metadata.json   : classe, hiperparâmetros, features, parâmetros do threshold dinâmico e bases;
      - <nome>_metrics.json    : métricas estatísticas, operacionais do threshold dinâmico, financeiras e gap.
    """
    import joblib
    from datetime import datetime

    dest = Path(diretorio_destino)
    dest.mkdir(parents=True, exist_ok=True)
    modelo = artefato['modelo']
    regra = artefato['regra_decisao']

    caminhos = {
        'artefato': dest / f'{nome_arquivo}.joblib',
        'metadata': dest / f'{nome_arquivo}_metadata.json',
        'metrics': dest / f'{nome_arquivo}_metrics.json',
    }

    joblib.dump(artefato, caminhos['artefato'])

    cols_bases = [c for c in df_metricas.columns if not c.startswith('Gap')]
    cols_gap   = [c for c in df_metricas.columns if c.startswith('Gap')]

    metadata = {
        'nome_modelo': artefato['nome_modelo'],
        'data_exportacao': datetime.now().isoformat(timespec='seconds'),
        'treinado_100pct_treino': bool(artefato.get('treinado_100pct_treino', True)),
        'n_amostras_treino_100pct': artefato.get('n_amostras_treino'),
        'classe': f"{modelo.__class__.__module__}.{modelo.__class__.__name__}",
        'calibracao': f"{artefato['calibrador'].__class__.__name__} (ajustada no OOF do Treino)",
        'hiperparametros': _serializar_params(modelo),
        'regra_decisao': regra,
        'total_features': len(artefato['features']),
        'features_utilizadas': artefato['features'],
        'bases_avaliadas_modelagem': {
            b: {'linhas': int(df_metricas.loc['Transações', b]), 'taxa_fraude': float(df_metricas.loc['Taxa Fraude', b])}
            for b in cols_bases
        },
        'uso': "artefato = joblib.load(...); utils.aplicar_regra_dinamica(artefato, X, valor_transacao_usd)",
    }
    with open(caminhos['metadata'], 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    def _extrair_bloco(col):
        s = df_metricas[col]
        return {
            'independentes_threshold': {
                'ROC-AUC': float(s['ROC-AUC']),
                'GINI': float(s['GINI']),
                'PR-AUC': float(s['PR-AUC']),
                'KS': float(s['KS']),
            },
            'regra_threshold_dinamico': {
                'Precision': float(s['Precision']),
                'Recall': float(s['Recall']),
                'F1-Score': float(s['F1-Score']),
                'Aprovacao_Legitimas': float(s['Aprovação']),
                'Aceitacao_Geral': float(s['Aceitação Geral']),
                'Qtd_VP': None if pd.isna(s['Qtd_VP']) else int(s['Qtd_VP']),
                'Qtd_FP': None if pd.isna(s['Qtd_FP']) else int(s['Qtd_FP']),
                'Qtd_FN': None if pd.isna(s['Qtd_FN']) else int(s['Qtd_FN']),
                'Qtd_VN': None if pd.isna(s['Qtd_VN']) else int(s['Qtd_VN']),
            },
            'financeiro_usd': {
                'Receita_VN_USD': None if pd.isna(s['Receita VN']) else float(s['Receita VN']),
                'Perda_FP_USD': None if pd.isna(s['Perda FP']) else float(s['Perda FP']),
                'Chargeback_FN_USD': None if pd.isna(s['Chargeback FN']) else float(s['Chargeback FN']),
                'Lucro_Liquido_USD': None if pd.isna(s['Lucro Líquido']) else float(s['Lucro Líquido']),
                'Lucro_Sem_Modelo_USD': None if pd.isna(s['Lucro Sem Modelo']) else float(s['Lucro Sem Modelo']),
                'Lucro_por_Transacao_USD': float(s['Lucro por Transação']),
                'Margem_sobre_Volume': float(s['Margem s/ Volume']),
            },
        }

    payload_metrics = {
        'nome_modelo': artefato['nome_modelo'],
        'treinado_100pct_treino': bool(artefato.get('treinado_100pct_treino', True)),
        'threshold_dinamico_params': regra,
        'metricas_por_base': {b: _extrair_bloco(b) for b in cols_bases},
        'gap_validacao_vs_cv': {
            k: float(df_metricas.loc[k, cols_gap[0]])
            for k in df_metricas.index
            if cols_gap and not pd.isna(df_metricas.loc[k, cols_gap[0]])
        },
    }
    with open(caminhos['metrics'], 'w', encoding='utf-8') as f:
        json.dump(payload_metrics, f, indent=2, ensure_ascii=False)

    print(f"Artefatos do modelo final ({artefato['nome_modelo']}) exportados com sucesso em {dest}:")
    for c in caminhos.values():
        print(f" - {c.name}")
    return caminhos


# ==============================================================================
# FUNÇÕES DE CLUSTERIZAÇÃO, REDUÇÃO DE DIMENSIONALIDADE E PERFILAMENTO DE FRAUDE
# ==============================================================================

def _salvar_figura_opcional(
    fig: Any,
    nome_arquivo: Optional[str] = None,
    save_dir: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> None:
    """Salva figura Matplotlib ou Plotly em disco apenas quando `nome_arquivo` e informado."""
    if not nome_arquivo:
        return
    base_dir = Path(save_dir) if save_dir is not None else (Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()) / "reports" / "figures"
    base_dir.mkdir(parents=True, exist_ok=True)
    caminho = base_dir / nome_arquivo
    if hasattr(fig, "write_html") and str(caminho).endswith(".html"):
        fig.write_html(str(caminho))
    elif hasattr(fig, "savefig"):
        fig.savefig(caminho, dpi=dpi, bbox_inches="tight")


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
    plt.show()
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
    plt.show()
    return evolucao_dia


# ==============================================================================
# CONSOLIDAÇÃO DOS ARQUÉTIPOS DE FRAUDE (SEÇÃO 6.0) E BENCHMARK HDBSCAN (SEÇÃO 7.0)
# ==============================================================================
# Paleta categórica validada (todos os pares, inclusive scatter) para os 4 arquétipos.
# Cada cor acompanha a entidade (arquétipo) em todos os gráficos, nunca o ranking.
PALETA_ARQUETIPOS_PADRAO: Tuple[str, ...] = ("#0B2545", "#13579B", "#0096C7", "#5DADE2")
COR_RUIDO_CLUSTER = "#97A3B0"
_COR_TEXTO_PRIMARIO = "#0B2545"
_COR_TEXTO_SECUNDARIO = "#5A6878"


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


# ==============================================================================
# AVALIAÇÃO OUT-OF-TIME (04_Avaliacao.ipynb): PERFORMANCE, DRIFT, SHAP E CLUSTERS
# ------------------------------------------------------------------------------
# Funções portadas do caderno 04_Model_Avaliation (Projeto_06 / src/plots.py,
# src/drift.py e src/model_utils.py) e adaptadas para:
#   - problema de FRAUDE (isFraud) com artefato calibrado + threshold dinâmico;
#   - comparação In-Time (Treino OOF + Validação) x Out-of-Time;
#   - identidade visual corporativa em paleta AZUL.
# ==============================================================================
from matplotlib.colors import LinearSegmentedColormap as _LinearSegmentedColormap

PALETA_AZUL: Dict[str, str] = {
    'marinho': '#0B2545',      # foco principal (OOT, fraude, destaque máximo)
    'principal': '#13579B',
    'medio': '#3E7CC4',
    'ciano': '#0096C7',        # destaque ciano corporativo
    'ciano_claro': '#48CAE4',  # apoio ciano claro
    'claro': '#8DB6E3',        # referência (In-Time)
    'gelo': '#E6F0FA',
    'cinza': '#97A3B0',        # legítimas / contexto
    'cinza_claro': '#D9DEE4',
    'texto': '#1B2633',
    'texto_sec': '#5A6878',
    'grade': '#EBEFF3',
    'acento': '#0096C7',       # destaque ciano para alertas (drift crítico, pior grupo)
}
COR_IN_TIME = PALETA_AZUL['claro']
COR_OOT = PALETA_AZUL['marinho']
COR_LEGITIMA = PALETA_AZUL['cinza']
COR_FRAUDE = PALETA_AZUL['principal']
CMAP_AZUL = _LinearSegmentedColormap.from_list('azul_corporativo', ['#F5F9FD', '#8DB6E3', '#13579B', '#0B2545'])
CMAP_AZUL_SHAP = _LinearSegmentedColormap.from_list('azul_shap', ['#A9C9EC', '#3E7CC4', '#0B2545'])
_CORES_PARTICAO = {'Treino (OOF)': '#FFFFFF', 'Validação': PALETA_AZUL['gelo'], 'OOT': '#D3E3F5'}


def aplicar_estilo_azul() -> None:
    """Aplica o tema visual corporativo (minimalista, paleta azul) em todos os gráficos Matplotlib/Seaborn."""
    from cycler import cycler
    p = PALETA_AZUL
    plt.rcParams.update({
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'figure.dpi': 110, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.edgecolor': p['cinza_claro'], 'axes.linewidth': 1.0,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'axes.axisbelow': True,
        'grid.color': p['grade'], 'grid.linewidth': 0.8, 'grid.linestyle': '-',
        'axes.titlesize': 13, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
        'axes.titlepad': 12, 'axes.titlecolor': p['texto'],
        'axes.labelsize': 10.5, 'axes.labelcolor': p['texto_sec'], 'axes.labelpad': 8,
        'xtick.color': p['texto_sec'], 'ytick.color': p['texto_sec'],
        'xtick.major.size': 0, 'ytick.major.size': 0,
        'legend.frameon': False, 'legend.fontsize': 9.5,
        'axes.prop_cycle': cycler(color=[p['marinho'], p['principal'], p['ciano'], p['claro'], p['cinza']]),
    })


def _titulo_ax(ax, titulo: str, subtitulo: Optional[str] = None) -> None:
    """Título à esquerda em negrito com subtítulo opcional em cinza logo abaixo."""
    ax.set_title(titulo, loc='left', pad=26 if subtitulo else 12)
    if subtitulo:
        ax.text(0, 1.02, subtitulo, transform=ax.transAxes, fontsize=9.5,
                color=PALETA_AZUL['texto_sec'], ha='left', va='bottom')


def _titulo_fig(fig, titulo: str, subtitulo: Optional[str] = None) -> None:
    fig.suptitle(titulo, x=0.01, ha='left', va='bottom', y=1.045 if subtitulo else 1.01,
                 fontsize=15, fontweight='bold', color=PALETA_AZUL['texto'])
    if subtitulo:
        fig.text(0.01, 1.01, subtitulo, ha='left', va='bottom', fontsize=10, color=PALETA_AZUL['texto_sec'])


def _salvar_fig(fig, filepath: Optional[Union[str, Path]] = None) -> None:
    """Salva a figura (quando `filepath` é informado) e exibe."""
    if filepath is not None:
        caminho = Path(filepath)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(caminho, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


def _status_psi(psi: float) -> str:
    if psi < 0.10:
        return 'Estável'
    if psi < 0.25:
        return 'Alerta'
    return 'Crítico'


# ------------------------------------------------------------------------------
# 1. ESCORAGEM DAS PARTIÇÕES (In-Time = Treino OOF + Validação | OOT)
# ------------------------------------------------------------------------------
def gerar_scores_oof_treino(artefato: Dict[str, Any], X_train: pd.DataFrame, y_train, groups, n_splits: int = 4) -> np.ndarray:
    """
    Probabilidade BRUTA Out-of-Fold no Treino com os MESMOS hiperparâmetros do artefato.

    O artefato foi ajustado em 100% do Treino: escorar o Treino com ele seria in-sample
    (GINI artificialmente alto). Reproduz o `GroupKFold` mensal do 02_Modelagem
    (mesmos grupos e n_splits do ajuste do calibrador), garantindo um score honesto.
    """
    from sklearn.base import clone

    X_ = X_train[artefato['features']]
    y_ = pd.Series(np.asarray(y_train)).reset_index(drop=True)
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    oof = np.zeros(len(X_), dtype=float)
    for i, (tr_idx, va_idx) in enumerate(gkf.split(X_, y_, groups=groups), start=1):
        m = clone(artefato['modelo'])
        m.fit(X_.iloc[tr_idx], y_.iloc[tr_idx])
        oof[va_idx] = m.predict_proba(X_.iloc[va_idx])[:, 1]
        print(f"  [OOF] fold {i}/{gkf.get_n_splits()} concluído | {len(va_idx):,} transações escoradas")
    return oof


def escorar_particao(
    artefato: Dict[str, Any], X: pd.DataFrame, df_ref: pd.DataFrame, particao: str, janela: str,
    prob_raw: Optional[np.ndarray] = None, col_target: str = 'isFraud', col_amt: str = 'TransactionAmt',
) -> pd.DataFrame:
    """
    Aplica o artefato final (prob. bruta -> calibrada -> threshold dinâmico por ticket) em uma partição.
    Se `prob_raw` for informado (ex.: OOF do Treino), ele substitui a predição do modelo final.
    Retorna um DataFrame por transação, alinhado ao índice de `df_ref`.
    """
    regra = artefato['regra_decisao']
    amt = obter_amt_real(df_ref, col_amt)
    if prob_raw is None:
        prob_raw = artefato['modelo'].predict_proba(X[artefato['features']])[:, 1]
    prob_raw = np.asarray(prob_raw, dtype=float)
    prob_cal = artefato['calibrador'].transform(prob_raw)
    th = threshold_dinamico(amt, regra['t_min'], regra['t_max'], regra['k'])
    ids = df_ref['TransactionID'].values if 'TransactionID' in df_ref.columns else df_ref.index.values
    return pd.DataFrame({
        'TransactionID': ids,
        'TransactionDT': df_ref['TransactionDT'].values,
        'Particao': particao,
        'Janela': janela,
        'isFraud': np.asarray(df_ref[col_target]).astype(int),
        'amt_usd': amt,
        'prob_raw': prob_raw,
        'prob_cal': prob_cal,
        'threshold': th,
        'bloquear': prob_cal >= th,
    }, index=df_ref.index)


def adicionar_bloco_temporal(df_scores: pd.DataFrame, dt_inicio: float, dias_por_bloco: int = 7,
                             nome_coluna: str = 'semana') -> pd.DataFrame:
    """Cria o índice do bloco temporal (ex.: semana) relativo ao início da base completa."""
    df = df_scores.copy()
    df[nome_coluna] = ((df['TransactionDT'] - dt_inicio) // (dias_por_bloco * 86400)).astype(int)
    return df


# ------------------------------------------------------------------------------
# 2. PERFORMANCE GLOBAL (TABELA DE MÉTRICAS + MATRIZES DE CONFUSÃO)
# ------------------------------------------------------------------------------
_LAYOUT_METRICAS_JANELAS = [
    # (chave, rótulo, tipo, delta comparável entre bases?)
    ('Transações', 'Transações', 'int', False),
    ('Taxa Fraude', 'Taxa de Fraude', 'pct', True),
    ('ROC-AUC', 'ROC-AUC', 'prob', True),
    ('GINI', 'GINI', 'prob', True),
    ('KS', 'KS', 'prob', True),
    ('PR-AUC', 'PR-AUC', 'prob', True),
    ('Precision', 'Precision', 'pct', True),
    ('Recall', 'Recall', 'pct', True),
    ('F1-Score', 'F1-Score', 'prob', True),
    ('Aprovação', 'Aprovação (Legítimas)', 'pct', True),
    ('Aceitação Geral', 'Aceitação Geral', 'pct', True),
    ('Qtd_VP', 'Fraudes Barradas (VP)', 'int', False),
    ('Qtd_FP', 'Legítimas Bloqueadas (FP)', 'int', False),
    ('Qtd_FN', 'Fraudes Aprovadas (FN)', 'int', False),
    ('Chargeback FN', 'Chargeback FN (US$)', 'usd', False),
    ('Lucro Líquido', 'Lucro Líquido (US$)', 'usd', False),
    ('Lucro Sem Modelo', 'Lucro Sem Modelo (US$)', 'usd', False),
    ('Lucro por Transação', 'Lucro por Transação (US$)', 'usd3', True),
    ('Margem s/ Volume', 'Margem s/ Volume', 'pct', True),
]


def tabela_metricas_janelas(bases: Dict[str, pd.DataFrame], ref: str = 'In-Time', alvo: str = 'OOT') -> pd.DataFrame:
    """
    Métricas de ranqueamento (prob. bruta), da regra dinâmica e financeiras para cada base
    (DataFrames gerados por `escorar_particao`) + coluna Δ (alvo − ref). Sem cross-validation:
    cada base é avaliada uma única vez com o artefato congelado.
    """
    metricas = {
        nome: _metricas_particao_unica(d['isFraud'], d['prob_raw'], d['bloquear'], d['amt_usd'])
        for nome, d in bases.items()
    }
    df = pd.DataFrame(metricas)
    df[f'Δ ({alvo} − {ref})'] = df[alvo] - df[ref]
    return df


def formatar_tabela_metricas(df_metricas: pd.DataFrame, destacar: str = 'OOT'):
    """Formata a tabela de `tabela_metricas_janelas` (tipos por linha) e destaca a coluna OOT em azul."""
    col_delta = [c for c in df_metricas.columns if c.startswith('Δ')]
    fmt = {'prob': '{:.4f}', 'pct': '{:.2%}', 'int': '{:,.0f}', 'usd': '$ {:,.0f}', 'usd3': '$ {:,.3f}'}
    fmt_d = {'prob': '{:+.4f}', 'pct': '{:+.2f} p.p.', 'usd3': '$ {:+,.3f}'}
    linhas = {}
    for chave, rotulo, tipo, tem_delta in _LAYOUT_METRICAS_JANELAS:
        if chave not in df_metricas.index:
            continue
        reg = {}
        for c in df_metricas.columns:
            v = df_metricas.loc[chave, c]
            if c in col_delta:
                if not tem_delta or pd.isna(v):
                    reg[c] = '—'
                else:
                    reg[c] = fmt_d.get(tipo, '{:+.4f}').format(v * 100 if tipo == 'pct' else v)
            else:
                reg[c] = fmt[tipo].format(v)
        linhas[rotulo] = reg
    df_fmt = pd.DataFrame(linhas).T
    df_fmt.index.name = 'Métrica'
    estilo = df_fmt.style.set_properties(**{'text-align': 'right'})
    if destacar in df_fmt.columns:
        estilo = estilo.set_properties(subset=[destacar], **{
            'background-color': PALETA_AZUL['gelo'], 'font-weight': 'bold', 'color': PALETA_AZUL['marinho']})
    return estilo


def plot_matrizes_confusao_janelas(bases: Dict[str, pd.DataFrame], figsize: Tuple[float, float] = (13, 5.2),
                                   filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Matrizes de confusão lado a lado (regra dinâmica) com % da classe real e contagens absolutas."""
    fig, axes = plt.subplots(1, len(bases), figsize=figsize)
    axes = np.atleast_1d(axes)
    registros = []
    nomes_q = [('VN', 'Legítima aprovada'), ('FP', 'Falso alarme'), ('FN', 'Fraude aprovada'), ('VP', 'Fraude barrada')]
    for ax, (nome, d) in zip(axes, bases.items()):
        cm = confusion_matrix(d['isFraud'], d['bloquear'].astype(int), labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        pct = np.array([[tn / max(1, tn + fp), fp / max(1, tn + fp)], [fn / max(1, fn + tp), tp / max(1, fn + tp)]])
        sns.heatmap(pct, cmap=CMAP_AZUL, vmin=0, vmax=1, cbar=False, linewidths=4, linecolor='white', ax=ax, annot=False)
        for i, (sig, desc) in enumerate(nomes_q):
            r, c = divmod(i, 2)
            n_abs = [tn, fp, fn, tp][i]
            cor = 'white' if pct[r, c] > 0.5 else PALETA_AZUL['texto']
            ax.text(c + 0.5, r + 0.42, f"{pct[r, c]:.1%}", ha='center', va='center', fontsize=17, fontweight='bold', color=cor)
            ax.text(c + 0.5, r + 0.66, f"{sig} · {desc}\nn = {n_abs:,}", ha='center', va='center', fontsize=9, color=cor)
            registros.append({'Base': nome, 'Classe Real': 'Fraude (1)' if r else 'Legítima (0)', 'Decisão': f"{sig} — {desc}",
                              'Transações': int(n_abs), '% da Classe': pct[r, c]})
        ax.set_xticklabels(['Aprovada', 'Bloqueada'], fontsize=10.5, fontweight='bold')
        ax.set_yticklabels(['Legítima', 'Fraude'], fontsize=10.5, fontweight='bold', rotation=0, va='center')
        ax.set_xlabel('Decisão do modelo (threshold dinâmico)')
        ax.set_ylabel('Classe real' if ax is axes[0] else '')
        ax.grid(False)
        _titulo_ax(ax, nome, f"{len(d):,} transações | fraude {d['isFraud'].mean():.2%}")
    _titulo_fig(fig, 'Matriz de Confusão — In-Time vs. Out-of-Time',
                'Percentual sobre a classe real (linhas somam 100%)')
    plt.tight_layout()
    _salvar_fig(fig, filepath)
    return pd.DataFrame(registros).set_index(['Base', 'Classe Real', 'Decisão'])


# ------------------------------------------------------------------------------
# 3. CAPACIDADE DE DISCRIMINAÇÃO (ROC / PR), SEPARAÇÃO (KDE) E KS
# ------------------------------------------------------------------------------
def plot_roc_pr_janelas(y_it, p_it, y_oot, p_oot, figsize: Tuple[float, float] = (15, 6),
                        filepath: Optional[Union[str, Path]] = None) -> Dict[str, float]:
    """
    Curvas ROC e Precision-Recall sobrepostas (In-Time em azul claro, OOT em azul marinho).
    Portada de `plot_curva_roc` / `plot_curva_pr` (Projeto_06) com a paleta corporativa azul.
    """
    from sklearn.metrics import roc_curve, precision_recall_curve
    y_it, y_oot = np.asarray(y_it), np.asarray(y_oot)
    fpr_i, tpr_i, _ = roc_curve(y_it, p_it)
    fpr_o, tpr_o, _ = roc_curve(y_oot, p_oot)
    auc_i, auc_o = roc_auc_score(y_it, p_it), roc_auc_score(y_oot, p_oot)
    pre_i, rec_i, _ = precision_recall_curve(y_it, p_it)
    pre_o, rec_o, _ = precision_recall_curve(y_oot, p_oot)
    ap_i, ap_o = average_precision_score(y_it, p_it), average_precision_score(y_oot, p_oot)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.plot([0, 1], [0, 1], ls='--', lw=1.2, color=PALETA_AZUL['cinza_claro'], label='Aleatório (AUC = 0.50)')
    ax1.plot(fpr_i, tpr_i, color=COR_IN_TIME, lw=2.6, label=f'In-Time   | AUC = {auc_i:.4f} · GINI = {2*auc_i-1:.4f}')
    ax1.plot(fpr_o, tpr_o, color=COR_OOT, lw=2.8, label=f'OOT        | AUC = {auc_o:.4f} · GINI = {2*auc_o-1:.4f}')
    ax1.fill_between(fpr_o, tpr_o, color=COR_OOT, alpha=0.05)
    ax1.set(xlim=(-0.01, 1.01), ylim=(-0.01, 1.02), xlabel='Taxa de Falsos Positivos (legítimas bloqueadas)',
            ylabel='Taxa de Verdadeiros Positivos (Recall)')
    ax1.legend(loc='lower right', fontsize=10)
    _titulo_ax(ax1, 'Curva ROC', f'ΔAUC (OOT − In-Time) = {auc_o - auc_i:+.4f}')

    ax2.plot(rec_i, pre_i, color=COR_IN_TIME, lw=2.4, label=f'In-Time | PR-AUC = {ap_i:.4f}')
    ax2.plot(rec_o, pre_o, color=COR_OOT, lw=2.6, label=f'OOT      | PR-AUC = {ap_o:.4f}')
    ax2.axhline(y_it.mean(), color=COR_IN_TIME, ls=':', lw=1.4, label=f'Prevalência In-Time ({y_it.mean():.2%})')
    ax2.axhline(y_oot.mean(), color=COR_OOT, ls=':', lw=1.4, label=f'Prevalência OOT ({y_oot.mean():.2%})')
    ax2.set(xlim=(-0.01, 1.01), ylim=(-0.01, 1.02), xlabel='Recall (fraudes capturadas)', ylabel='Precision (acerto no bloqueio)')
    ax2.legend(loc='upper right', fontsize=9.5)
    _titulo_ax(ax2, 'Curva Precision-Recall', 'Mais sensível ao desbalanceamento — a linha pontilhada é o modelo aleatório')

    _titulo_fig(fig, 'Estabilidade da Discriminação — In-Time vs. Out-of-Time')
    plt.tight_layout()
    _salvar_fig(fig, filepath)
    return {'auc_in_time': float(auc_i), 'auc_oot': float(auc_o), 'pr_auc_in_time': float(ap_i), 'pr_auc_oot': float(ap_o)}


def plot_kde_scores_janelas(y_it, p_it, y_oot, p_oot, faixa_corte: Optional[Tuple[float, float]] = None,
                            calibrador=None, amostra_max: int = 60000, random_state: int = 42,
                            figsize: Tuple[float, float] = (16, 5.8), filepath: Optional[Union[str, Path]] = None) -> None:
    """
    Separação de risco (KDE) Legítima x Fraude em cada janela (portada de `plot_distribuicao_kde_scores`).
    `faixa_corte` = (corte p/ alto ticket, corte p/ micro-compra) na MESMA escala do score plotado:
    a regra dinâmica varia com o valor, então o corte é uma faixa sombreada, não uma linha.
    Amostragem estratificada por classe (`amostra_max`) apenas para acelerar o KDE.
    `calibrador` (isotônica): se informado, adiciona um eixo X superior com a probabilidade CALIBRADA
    equivalente a cada posição do score bruto (é nessa escala que t_min/t_max da regra são definidos).
    """
    rng = np.random.default_rng(random_state)
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for ax, (y, p, nome) in zip(axes, [(y_it, p_it, 'In-Time'), (y_oot, p_oot, 'Out-of-Time (OOT)')]):
        y, p = np.asarray(y), np.asarray(p, dtype=float)
        for classe, cor, rotulo, alpha, lw in [(0, COR_LEGITIMA, 'Legítima (0)', 0.18, 2.0), (1, COR_FRAUDE, 'Fraude (1)', 0.30, 2.6)]:
            s = p[y == classe]
            if len(s) > amostra_max:
                s = rng.choice(s, amostra_max, replace=False)
            sns.kdeplot(x=s, ax=ax, color=cor, fill=True, alpha=alpha, lw=lw, clip=(0, 1), cut=0,
                        label=f'{rotulo} · mediana {np.median(p[y == classe]):.3f}')
        if faixa_corte is not None:
            lo, hi = faixa_corte
            ax.axvspan(lo, hi, color=PALETA_AZUL['acento'], alpha=0.12, lw=0, label=f'Faixa do corte dinâmico [{lo:.3f} – {hi:.3f}]')
            ax.axvline(lo, color=PALETA_AZUL['acento'], lw=1.0, ls='--', alpha=0.8)
            ax.axvline(hi, color=PALETA_AZUL['acento'], lw=1.0, ls='--', alpha=0.8)
        ks = ks_2samp(p[y == 1], p[y == 0]).statistic
        ax.set(xlim=(0, 1), xlabel='Score do modelo (probabilidade bruta)')
        ax.set_ylabel('Densidade' if ax is axes[0] else '')
        ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
        ax.legend(loc='upper right', fontsize=9.5, frameon=True, facecolor='white', edgecolor='white', framealpha=0.9)
        if calibrador is not None:
            ax_cal = ax.twiny()
            ax_cal.set_xlim(ax.get_xlim())
            ticks_raw = np.linspace(0, 1, 6)
            fmt_cal = lambda v: f'{v:.1%}' if v < 0.10 else f'{v:.0%}'
            rotulos = [fmt_cal(v) for v in np.clip(calibrador.predict(ticks_raw), 0, 1)]
            if faixa_corte is not None:
                # Um único rótulo para a faixa (t_min – t_max) evita sobreposição quando ela é estreita
                centro = float(np.mean(faixa_corte))
                manter = np.abs(ticks_raw - centro) > 0.12
                lo_cal, hi_cal = np.clip(calibrador.predict(np.asarray(faixa_corte)), 0, 1)
                ticks_raw = np.r_[ticks_raw[manter], centro]
                rotulos = [r for r, m in zip(rotulos, manter) if m] + [f'corte {fmt_cal(lo_cal)}–{fmt_cal(hi_cal)}']
            ax_cal.set_xticks(ticks_raw)
            ax_cal.set_xticklabels(rotulos, fontsize=8.5, color=PALETA_AZUL['texto_sec'])
            if faixa_corte is not None:
                lbl = ax_cal.get_xticklabels()[-1]
                lbl.set_color(PALETA_AZUL['acento'])
                lbl.set_fontweight('bold')
            ax_cal.set_xlabel('Probabilidade calibrada equivalente (escala real de fraude)', fontsize=9,
                              color=PALETA_AZUL['texto_sec'], labelpad=6)
            ax_cal.grid(False)
            ax_cal.tick_params(axis='x', length=3, pad=2)
            # Título/subtítulo acima do eixo calibrado (sem sobrepor os rótulos do eixo superior)
            ax.set_title(nome, loc='left', pad=58)
            ax.annotate(f'{len(p):,} transações | KS = {ks:.4f}', xy=(0, 1), xycoords='axes fraction',
                        xytext=(0, 40), textcoords='offset points', fontsize=9.5,
                        color=PALETA_AZUL['texto_sec'], ha='left', va='bottom')
        else:
            _titulo_ax(ax, nome, f'{len(p):,} transações | KS = {ks:.4f}')
    _titulo_fig(fig, 'Separação de Risco (KDE) — Distribuição dos Scores por Classe',
                'Quanto menor a sobreposição entre as áreas, melhor o modelo separa fraude de legítima')
    plt.tight_layout()
    _salvar_fig(fig, filepath)


def plot_cdf_ks_janelas(y_it, p_it, y_oot, p_oot, figsize: Tuple[float, float] = (16, 5.8),
                        filepath: Optional[Union[str, Path]] = None) -> Tuple[float, float]:
    """
    CDFs empíricas Legítima x Fraude e ponto de separação máxima (KS) em cada janela.
    Portada de `plot_cdf_ks`. Retorna (KS In-Time, KS OOT) em escala 0–1.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    ks_res = []
    for ax, (y, p, nome) in zip(axes, [(y_it, p_it, 'In-Time'), (y_oot, p_oot, 'Out-of-Time (OOT)')]):
        y, p = np.asarray(y), np.asarray(p, dtype=float)
        s0, s1 = np.sort(p[y == 0]), np.sort(p[y == 1])
        grade = np.unique(p)
        cdf0 = np.searchsorted(s0, grade, side='right') / len(s0)
        cdf1 = np.searchsorted(s1, grade, side='right') / len(s1)
        diff = np.abs(cdf0 - cdf1)
        i_ks = int(np.argmax(diff))
        x_ks, ks = grade[i_ks], float(diff[i_ks])
        ks_res.append(ks)
        ax.plot(grade, cdf0, color=COR_LEGITIMA, lw=2.4, label='CDF Legítima (0)')
        ax.plot(grade, cdf1, color=COR_FRAUDE, lw=2.6, label='CDF Fraude (1)')
        ax.fill_between(grade, cdf1, cdf0, where=cdf0 >= cdf1, color=PALETA_AZUL['claro'], alpha=0.15)
        ax.vlines(x_ks, min(cdf0[i_ks], cdf1[i_ks]), max(cdf0[i_ks], cdf1[i_ks]), color=COR_OOT, lw=2.2, ls='--',
                  label=f'KS = {ks:.4f}')
        ax.plot([x_ks, x_ks], [cdf0[i_ks], cdf1[i_ks]], 'o', color=COR_OOT, ms=6)
        ax.annotate(f'KS máx. = {ks:.1%}\nscore = {x_ks:.3f}', xy=(x_ks, (cdf0[i_ks] + cdf1[i_ks]) / 2),
                    xytext=(min(x_ks + 0.12, 0.75), 0.35), fontsize=10, fontweight='bold', color=COR_OOT,
                    arrowprops=dict(arrowstyle='-|>', color=COR_OOT, lw=1.1, connectionstyle='arc3,rad=.2'),
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=PALETA_AZUL['cinza_claro']))
        ax.set(xlim=(0, 1), ylim=(-0.01, 1.02), xlabel='Score do modelo (probabilidade bruta)')
        ax.set_ylabel('Proporção acumulada (CDF)' if ax is axes[0] else '')
        ax.legend(loc='lower right', fontsize=10)
        _titulo_ax(ax, nome)
    _titulo_fig(fig, 'Kolmogorov-Smirnov (KS) — Separação Máxima entre Legítimas e Fraudes',
                f'ΔKS (OOT − In-Time) = {ks_res[1] - ks_res[0]:+.4f}')
    plt.tight_layout()
    _salvar_fig(fig, filepath)
    return ks_res[0], ks_res[1]


def gerar_tabela_decis_ks(y_true, prob, n_decis: int = 10) -> pd.DataFrame:
    """
    Tabela de ordenação de risco por decis de score (Decil 10 = maior risco), portada do Projeto_06:
    taxa de fraude, captura acumulada de fraudes a partir do topo, KS acumulado e Lift.
    """
    df = pd.DataFrame({'y': np.asarray(y_true).astype(int), 'score': np.asarray(prob, dtype=float)})
    df['decil'] = pd.qcut(df['score'].rank(method='first'), q=n_decis, labels=False) + 1
    tab = df.groupby('decil').agg(transacoes=('y', 'size'), fraudes=('y', 'sum'),
                                  score_min=('score', 'min'), score_medio=('score', 'mean'),
                                  score_max=('score', 'max')).reset_index()
    tab['legitimas'] = tab['transacoes'] - tab['fraudes']
    tab['taxa_fraude_pct'] = tab['fraudes'] / tab['transacoes'] * 100
    tab = tab.sort_values('decil', ascending=False).reset_index(drop=True)
    tab['captura_fraudes_acum_pct'] = tab['fraudes'].cumsum() / tab['fraudes'].sum() * 100
    tab['legitimas_acum_pct'] = tab['legitimas'].cumsum() / tab['legitimas'].sum() * 100
    tab['ks_pct'] = (tab['captura_fraudes_acum_pct'] - tab['legitimas_acum_pct']).abs()
    tab['lift'] = tab['taxa_fraude_pct'] / (df['y'].mean() * 100)
    return tab.sort_values('decil').reset_index(drop=True)


def plot_decis_janelas(tab_it: pd.DataFrame, tab_oot: pd.DataFrame, figsize: Tuple[float, float] = (16, 5.6),
                       filepath: Optional[Union[str, Path]] = None) -> None:
    """Escada de risco (taxa de fraude por decil) e curva de ganho (captura acumulada) In-Time x OOT."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    x = tab_it['decil'].values
    w = 0.38
    b1 = ax1.bar(x - w / 2, tab_it['taxa_fraude_pct'], width=w, color=COR_IN_TIME, label='In-Time')
    b2 = ax1.bar(x + w / 2, tab_oot['taxa_fraude_pct'], width=w, color=COR_OOT, label='OOT')
    for bar, v in zip(b2, tab_oot['taxa_fraude_pct']):
        ax1.text(bar.get_x() + bar.get_width() / 2, v, f'{v:.1f}%', ha='center', va='bottom', fontsize=8.5,
                 fontweight='bold', color=COR_OOT)
    ax1.set(xticks=x, xlabel='Decil de score (10 = maior risco)', ylabel='Taxa de fraude (%)')
    ax1.grid(axis='x', visible=False)
    ax1.legend(loc='upper left')
    _titulo_ax(ax1, 'Escada de Risco por Decil', 'Monotonicidade = o modelo ordena o risco corretamente')

    pct_pop = np.arange(1, len(x) + 1) * 100 / len(x)
    cap_it = tab_it.sort_values('decil', ascending=False)['captura_fraudes_acum_pct'].values
    cap_oot = tab_oot.sort_values('decil', ascending=False)['captura_fraudes_acum_pct'].values
    ax2.plot([0, 100], [0, 100], ls='--', color=PALETA_AZUL['cinza_claro'], lw=1.2, label='Aleatório')
    ax2.plot(np.r_[0, pct_pop], np.r_[0, cap_it], color=COR_IN_TIME, lw=2.4, marker='o', ms=5, label='In-Time')
    ax2.plot(np.r_[0, pct_pop], np.r_[0, cap_oot], color=COR_OOT, lw=2.6, marker='o', ms=5, label='OOT')
    ax2.annotate(f'Top 10%: {cap_oot[0]:.1f}% das fraudes', xy=(10, cap_oot[0]), xytext=(22, cap_oot[0] - 18),
                 fontsize=10, fontweight='bold', color=COR_OOT,
                 arrowprops=dict(arrowstyle='-|>', color=COR_OOT, lw=1.1))
    ax2.set(xlim=(0, 101), ylim=(0, 102), xlabel='% da população (ordenada do maior para o menor score)',
            ylabel='% das fraudes capturadas')
    ax2.legend(loc='lower right')
    _titulo_ax(ax2, 'Curva de Ganho (Captura Acumulada)', 'Quanto mais "colada" ao canto superior esquerdo, melhor')
    plt.tight_layout()
    _salvar_fig(fig, filepath)


# ------------------------------------------------------------------------------
# 4. INTERPRETABILIDADE (SHAP)
# ------------------------------------------------------------------------------
def calcular_shap_lgbm(modelo, X: pd.DataFrame, n_amostra: int = 20000, random_state: int = 42) -> Tuple[np.ndarray, pd.DataFrame]:
    """SHAP values (classe positiva, log-odds) via TreeExplainer em uma amostra de `X`."""
    if shap is None:
        raise ImportError("Pacote 'shap' não encontrado. Instale com `pip install shap`.")
    Xs = X.sample(n=min(n_amostra, len(X)), random_state=random_state)
    sv = shap.TreeExplainer(modelo).shap_values(Xs)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv, Xs


def ranking_shap(shap_values: np.ndarray, features: Iterable[str]) -> pd.DataFrame:
    """Ranking global por |SHAP| médio (da variável mais para a menos relevante)."""
    df = pd.DataFrame({'Feature': list(features), 'SHAP_medio_abs': np.abs(shap_values).mean(axis=0)})
    df = df.sort_values('SHAP_medio_abs', ascending=False).reset_index(drop=True)
    df['Rank'] = np.arange(1, len(df) + 1)
    df['%_Impacto'] = df['SHAP_medio_abs'] / df['SHAP_medio_abs'].sum() * 100
    return df


def plot_importancia_shap(df_rank: pd.DataFrame, top_n: int = 15, n_destaque: int = 3,
                          titulo: str = 'Impacto Global das Variáveis (|SHAP| médio)', subtitulo: Optional[str] = None,
                          figsize: Tuple[float, float] = (11, 7.5), filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Ranking horizontal de |SHAP|: Top `n_destaque` em azul marinho, demais em azul claro (portada de `plot_importancia_shap`)."""
    top = df_rank.head(top_n).iloc[::-1]
    cores = [COR_OOT if i >= len(top) - n_destaque else PALETA_AZUL['claro'] for i in range(len(top))]
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(top['Feature'], top['SHAP_medio_abs'], color=cores, height=0.68)
    ax.bar_label(bars, fmt='%.4f', padding=5, fontsize=9.5, fontweight='bold', color=PALETA_AZUL['texto'])
    ax.set_xlim(0, top['SHAP_medio_abs'].max() * 1.18)
    ax.set_xlabel('Impacto médio na predição (|SHAP|, log-odds)')
    ax.grid(axis='y', visible=False)
    _titulo_ax(ax, titulo, subtitulo)
    plt.tight_layout()
    _salvar_fig(fig, filepath)
    return df_rank.head(top_n)


def plot_shap_beeswarm(shap_values: np.ndarray, X_amostra: pd.DataFrame, top_n: int = 15,
                       titulo: str = 'Direção do Impacto (SHAP Beeswarm)', filepath: Optional[Union[str, Path]] = None) -> None:
    """Beeswarm SHAP com colormap azul (claro = valor baixo da feature, marinho = valor alto)."""
    shap.summary_plot(shap_values, X_amostra, max_display=top_n, cmap=CMAP_AZUL_SHAP, show=False,
                      plot_size=(11, 0.45 * top_n + 1.8), color_bar_label='Valor da feature')
    fig = plt.gcf()
    ax = fig.axes[0]
    if len(fig.axes) > 1:
        try:
            fig.axes[-1].set_yticklabels(['Baixo', 'Alto'])
        except Exception:
            pass
    ax.set_xlabel('SHAP value (impacto no log-odds de fraude)')
    _titulo_ax(ax, titulo, 'Cada ponto é uma transação · à direita do zero = empurra para FRAUDE')
    _salvar_fig(fig, filepath)


def plot_shap_janelas(df_rank_it: pd.DataFrame, df_rank_oot: pd.DataFrame, top_n: int = 15,
                      figsize: Tuple[float, float] = (11, 7.5), filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Dumbbell In-Time x OOT do |SHAP| médio (top N do OOT). Mostra se os motores de decisão
    do modelo continuam os mesmos fora do tempo. Retorna a tabela comparativa com a correlação de Spearman.
    """
    m = df_rank_oot[['Feature', 'SHAP_medio_abs', 'Rank']].merge(
        df_rank_it[['Feature', 'SHAP_medio_abs', 'Rank']], on='Feature', suffixes=('_OOT', '_InTime'))
    m['Δ_Rank'] = m['Rank_InTime'] - m['Rank_OOT']
    m['Δ_%'] = (m['SHAP_medio_abs_OOT'] / m['SHAP_medio_abs_InTime'].replace(0, np.nan) - 1) * 100
    rho = stats.spearmanr(m['SHAP_medio_abs_InTime'], m['SHAP_medio_abs_OOT']).correlation
    top = m.sort_values('SHAP_medio_abs_OOT', ascending=False).head(top_n).iloc[::-1]
    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=figsize)
    ax.hlines(y, top['SHAP_medio_abs_InTime'], top['SHAP_medio_abs_OOT'], color=PALETA_AZUL['cinza_claro'], lw=3, zorder=1)
    ax.scatter(top['SHAP_medio_abs_InTime'], y, s=70, color=COR_IN_TIME, zorder=2, label='In-Time')
    ax.scatter(top['SHAP_medio_abs_OOT'], y, s=80, color=COR_OOT, zorder=3, label='OOT')
    for yi, (_, r) in zip(y, top.iterrows()):
        if abs(r['Δ_Rank']) >= 3:
            ax.text(max(r['SHAP_medio_abs_OOT'], r['SHAP_medio_abs_InTime']) * 1.03, yi,
                    f"rank {int(r['Rank_InTime'])}→{int(r['Rank_OOT'])}", va='center', fontsize=8.5,
                    color=PALETA_AZUL['acento'], fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(top['Feature'])
    ax.set_xlabel('|SHAP| médio')
    ax.grid(axis='y', visible=False)
    ax.legend(loc='lower right')
    _titulo_ax(ax, 'Estabilidade dos Motores de Decisão — |SHAP| In-Time vs. OOT',
               f'Correlação de Spearman entre os rankings = {rho:.3f} · em laranja: variáveis que mudaram ≥ 3 posições')
    plt.tight_layout()
    _salvar_fig(fig, filepath)
    m.attrs['spearman'] = float(rho)
    return m.sort_values('Rank_OOT').reset_index(drop=True)


# ------------------------------------------------------------------------------
# 5. DRIFT: TARGET, PSI DO SCORE, PSI DAS FEATURES E ESTABILIDADE TEMPORAL
# ------------------------------------------------------------------------------
def tabela_temporal(df_scores: pd.DataFrame, col_bloco: str = 'semana', min_fraudes: int = 30,
                    min_transacoes: int = 1000) -> pd.DataFrame:
    """
    Métricas por bloco temporal (ex.: semana): volume, taxa de fraude, ROC-AUC, GINI, KS, PR-AUC,
    Recall, Precision e Aprovação. Partição/Janela do bloco = a majoritária (blocos de fronteira).
    Blocos com poucas fraudes/transações recebem NaN nas métricas de ranqueamento.
    """
    linhas = []
    for b, d in df_scores.groupby(col_bloco, sort=True):
        y, p = d['isFraud'].values, d['prob_raw'].values
        n_f = int(y.sum())
        valido = (n_f >= min_fraudes) and (len(d) >= min_transacoes) and (n_f < len(d))
        auc = roc_auc_score(y, p) if valido else np.nan
        r = calcular_score_financeiro(y, d['bloquear'].values, d['amt_usd'].values)
        linhas.append({
            col_bloco: int(b),
            'Particao': d['Particao'].value_counts().idxmax(),
            'Janela': d['Janela'].value_counts().idxmax(),
            'Transacoes': len(d),
            'Fraudes': n_f,
            'Taxa_Fraude_%': y.mean() * 100,
            'ROC_AUC': auc,
            'GINI': 2 * auc - 1 if valido else np.nan,
            'KS': ks_2samp(p[y == 1], p[y == 0]).statistic if valido else np.nan,
            'PR_AUC': average_precision_score(y, p) if valido else np.nan,
            'Recall_%': r['Recall'] * 100,
            'Precision_%': r['Precision'] * 100,
            'Aprovacao_%': r['Approval_Rate'] * 100,
            'Lucro_USD': r['Resultado_Liquido'],
        })
    return pd.DataFrame(linhas)


def _sombrear_particoes(ax, df_temp: pd.DataFrame, col_bloco: str) -> None:
    """Fundo por partição (Validação = gelo, OOT = azul suave) com rótulo no topo do gráfico."""
    for part, cor in _CORES_PARTICAO.items():
        blocos = df_temp.loc[df_temp['Particao'] == part, col_bloco]
        if blocos.empty:
            continue
        x0, x1 = blocos.min() - 0.5, blocos.max() + 0.5
        if part != 'Treino (OOF)':
            ax.axvspan(x0, x1, color=cor, alpha=0.6, lw=0, zorder=0)
        ax.text((x0 + x1) / 2, 0.97, part.upper(), transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=9, fontweight='bold', color=COR_OOT if part == 'OOT' else PALETA_AZUL['texto_sec'])


def _linhas_media_janela(ax, df_temp: pd.DataFrame, col_bloco: str, col_valor: str, fmt: str) -> Dict[str, float]:
    medias = {}
    for jan, cor in [('In-Time', PALETA_AZUL['medio']), ('OOT', COR_OOT)]:
        d = df_temp[df_temp['Janela'] == jan].dropna(subset=[col_valor])
        if d.empty:
            continue
        med = float(np.average(d[col_valor], weights=d['Transacoes']))
        medias[jan] = med
        ax.hlines(med, d[col_bloco].min() - 0.4, d[col_bloco].max() + 0.4, colors=cor, linestyles='--', lw=1.6, zorder=2)
        caixa = dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='none', alpha=0.85)
        if jan == 'OOT':
            ax.annotate(f'média {jan}\n{fmt.format(med)}', xy=(d[col_bloco].max() + 0.4, med), xytext=(10, 0),
                        textcoords='offset points', va='center', ha='left', fontsize=9, fontweight='bold',
                        color=cor, bbox=caixa, zorder=5)
        else:
            ax.annotate(f'média {jan}: {fmt.format(med)}', xy=(d[col_bloco].min() - 0.4, med), xytext=(2, 9),
                        textcoords='offset points', va='bottom', ha='left', fontsize=9, fontweight='bold',
                        color=cor, bbox=caixa, zorder=2.5)
    return medias


def plot_target_drift_temporal(df_temp: pd.DataFrame, col_bloco: str = 'semana', rotulo_bloco: str = 'Semana',
                               figsize: Tuple[float, float] = (15, 5.8), filepath: Optional[Union[str, Path]] = None) -> Dict[str, float]:
    """
    Target drift: taxa de fraude (%) por bloco temporal ao longo de Treino → Validação → OOT
    (portada de `plot_target_drift_mensal`), com volume em barras ao fundo e médias por janela.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax_vol = ax.twinx()
    ax_vol.bar(df_temp[col_bloco], df_temp['Transacoes'], width=0.75, color=PALETA_AZUL['cinza_claro'], alpha=0.45, zorder=0)
    ax_vol.set_ylim(0, df_temp['Transacoes'].max() * 3.2)
    ax_vol.set_yticks([])
    ax_vol.grid(False)
    for s in ax_vol.spines.values():
        s.set_visible(False)
    ax.set_zorder(ax_vol.get_zorder() + 1)
    ax.patch.set_visible(False)

    _sombrear_particoes(ax, df_temp, col_bloco)
    ax.plot(df_temp[col_bloco], df_temp['Taxa_Fraude_%'], color=COR_OOT, lw=2.6, marker='o', ms=6, zorder=3,
            markerfacecolor='white', markeredgewidth=2)
    medias = _linhas_media_janela(ax, df_temp, col_bloco, 'Taxa_Fraude_%', '{:.2f}%')
    i_max = df_temp['Taxa_Fraude_%'].idxmax()
    ax.annotate(f"pico {df_temp.loc[i_max, 'Taxa_Fraude_%']:.2f}%",
                xy=(df_temp.loc[i_max, col_bloco], df_temp.loc[i_max, 'Taxa_Fraude_%']), xytext=(0, 10),
                textcoords='offset points', ha='center', fontsize=9, fontweight='bold', color=COR_OOT)
    ax.set_xlabel(f'{rotulo_bloco} (desde o início da base) · barras cinza = volume de transações')
    ax.set_ylabel('Taxa de fraude (%)')
    ax.set_xlim(df_temp[col_bloco].min() - 0.6, df_temp[col_bloco].max() + 2.6)
    ymax = df_temp['Taxa_Fraude_%'].max()
    ax.set_ylim(0, ymax * 1.30)
    sub = (f"Variação da média OOT vs. In-Time: {medias['OOT'] - medias['In-Time']:+.2f} p.p. "
           f"({(medias['OOT'] / medias['In-Time'] - 1) * 100:+.1f}% relativo)") if len(medias) == 2 else None
    _titulo_ax(ax, 'Target Drift — Taxa de Fraude ao Longo do Tempo', sub)
    plt.tight_layout()
    _salvar_fig(fig, filepath)
    return medias


def teste_qui_quadrado_temporal(tabela: pd.DataFrame, col_eventos: str = 'Fraudes', col_volume: str = 'Transacoes',
                                rotulo: str = '') -> Dict[str, float]:
    """
    Qui-quadrado de independência (período x target) + V de Cramér como tamanho do efeito.
    Portada de `src/drift.py` recebendo contagens diretamente (eventos e volume por linha).
    """
    cont = pd.DataFrame({'eventos': tabela[col_eventos].astype(int),
                         'nao_eventos': (tabela[col_volume] - tabela[col_eventos]).astype(int)})
    chi2, p, gl, esperado = stats.chi2_contingency(cont.values)
    n = cont.values.sum()
    k = min(cont.shape) - 1
    v = float(np.sqrt(chi2 / (n * k))) if n * k > 0 else float('nan')
    forca = 'desprezível' if v < 0.05 else ('fraco' if v < 0.10 else ('moderado' if v < 0.30 else 'forte'))
    print(f"[Qui²{' — ' + rotulo if rotulo else ''}] χ² = {chi2:,.2f} | gl = {gl} | p-valor = {p:.2e} | "
          f"V de Cramér = {v:.4f} (efeito {forca})")
    return {'chi2': float(chi2), 'p_valor': float(p), 'gl': int(gl), 'freq_esperada_min': float(esperado.min()),
            'cramers_v': v, 'efeito': forca}


def calcular_psi_scores(score_ref, score_atual, bins: int = 10) -> Tuple[pd.DataFrame, float]:
    """
    PSI do SCORE com faixas pelos decis da referência (In-Time). Portada de `src/drift.py`.
    Retorna a tabela por faixa (com a contribuição de cada faixa) e o PSI total.
    """
    score_ref = pd.Series(np.asarray(score_ref, dtype=float))
    score_atual = pd.Series(np.asarray(score_atual, dtype=float))
    limites = np.unique(np.quantile(score_ref, np.linspace(0, 1, bins + 1)))
    limites[0], limites[-1] = -np.inf, np.inf
    ref = pd.cut(score_ref, bins=limites).value_counts(sort=False) / len(score_ref)
    atu = pd.cut(score_atual, bins=limites).value_counts(sort=False) / len(score_atual)
    df = pd.DataFrame({'Faixa': [f'D{i + 1}' for i in range(len(ref))],
                       'Limite_Sup': limites[1:], 'InTime_%': ref.values, 'OOT_%': atu.values})
    df[['InTime_%', 'OOT_%']] = df[['InTime_%', 'OOT_%']].clip(lower=1e-4)
    df['PSI'] = (df['OOT_%'] - df['InTime_%']) * np.log(df['OOT_%'] / df['InTime_%'])
    psi = float(df['PSI'].sum())
    print(f"[PSI do Score] PSI = {psi:.4f} → {_status_psi(psi)}")
    return df, psi


def plot_psi_score_faixas(df_psi: pd.DataFrame, psi_total: float, figsize: Tuple[float, float] = (13, 5.2),
                          rotulo_atual: str = 'OOT',
                          filepath: Optional[Union[str, Path]] = None) -> None:
    """Distribuição da população por faixa de score (decis In-Time): In-Time x janela atual, com o PSI de cada faixa."""
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(df_psi))
    w = 0.38
    ax.bar(x - w / 2, df_psi['InTime_%'] * 100, width=w, color=COR_IN_TIME, label='In-Time')
    ax.bar(x + w / 2, df_psi['OOT_%'] * 100, width=w, color=COR_OOT, label=rotulo_atual)
    topo = max(df_psi['InTime_%'].max(), df_psi['OOT_%'].max()) * 100
    y_rot = topo * 1.12
    for xi, psi_f in zip(x, df_psi['PSI']):
        ax.text(xi, y_rot, f'{psi_f:.4f}', ha='center', va='bottom', fontsize=8.5,
                color=PALETA_AZUL['acento'] if psi_f > 0.01 else PALETA_AZUL['texto_sec'], fontweight='bold')
    ax.text(-0.55, y_rot, 'PSI faixa →', ha='right', va='bottom', fontsize=8.5, color=PALETA_AZUL['texto_sec'])
    ax.set(xticks=x, xticklabels=df_psi['Faixa'], ylim=(0, topo * 1.45), xlim=(-1.3, len(x) - 0.4),
           xlabel='Faixa de score (decis da janela In-Time · D10 = maior risco)', ylabel='% da população')
    ax.grid(axis='x', visible=False)
    ax.legend(loc='upper center', ncol=2, frameon=True, facecolor='white', edgecolor='white', framealpha=0.9)
    _titulo_ax(ax, f'PSI do Score — {psi_total:.4f} ({_status_psi(psi_total)})',
               f'Por construção, cada decil In-Time tem 10% da população; desvios de {rotulo_atual} indicam mudança no perfil de risco')
    plt.tight_layout()
    _salvar_fig(fig, filepath)


def relatorio_psi_features(X_ref: pd.DataFrame, X_atual: pd.DataFrame, features: Optional[List[str]] = None,
                           bins: int = 10) -> pd.DataFrame:
    """
    PSI e KS de deslocamento para TODAS as features numéricas (inclui float32/int32 do `safe_downcast`).
    Retorna valores numéricos (não strings) com o status regulatório.
    """
    features = features or list(X_ref.select_dtypes(include=[np.number]).columns)
    linhas = []
    for col in features:
        a, b = X_ref[col].astype('float64'), X_atual[col].astype('float64')
        psi = float(calcular_psi(a, b, bins=bins))
        a_, b_ = a.dropna(), b.dropna()
        ks = float(ks_2samp(a_, b_).statistic) if len(a_) and len(b_) else np.nan
        linhas.append({'Feature': col, 'PSI': psi, 'KS_shift': ks, 'Status': _status_psi(psi),
                       '%_Nulos_InTime': a.isna().mean() * 100, '%_Nulos_OOT': b.isna().mean() * 100})
    return pd.DataFrame(linhas).sort_values('PSI', ascending=False).reset_index(drop=True)


def plot_psi_vs_shap(df_psi: pd.DataFrame, df_rank_shap: pd.DataFrame, n_rotulos: int = 8,
                     figsize: Tuple[float, float] = (13, 6.5), filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Matriz Drift x Importância: eixo X = |SHAP| médio, eixo Y = PSI.
    O quadrante superior direito (drift alto em variável importante) é o risco real para o modelo.
    Retorna a tabela combinada com o `Risco_Drift` = PSI × %Impacto SHAP.
    """
    m = df_psi.merge(df_rank_shap[['Feature', 'SHAP_medio_abs', '%_Impacto', 'Rank']], on='Feature', how='inner')
    m['Risco_Drift'] = m['PSI'] * m['%_Impacto']
    m = m.sort_values('Risco_Drift', ascending=False).reset_index(drop=True)
    cores = m['Status'].map({'Estável': PALETA_AZUL['claro'], 'Alerta': PALETA_AZUL['principal'],
                             'Crítico': PALETA_AZUL['acento']})
    fig, ax = plt.subplots(figsize=figsize)
    ax.axhspan(0.25, max(0.3, m['PSI'].max() * 1.15), color=PALETA_AZUL['acento'], alpha=0.06, lw=0)
    ax.axhspan(0.10, 0.25, color=PALETA_AZUL['gelo'], alpha=0.9, lw=0)
    ax.axhline(0.10, color=PALETA_AZUL['principal'], ls='--', lw=1)
    ax.axhline(0.25, color=PALETA_AZUL['acento'], ls='--', lw=1)
    ax.scatter(m['SHAP_medio_abs'], m['PSI'], s=40 + 600 * m['%_Impacto'] / m['%_Impacto'].max(), c=cores,
               edgecolor='white', lw=1.2, zorder=3, alpha=0.95)
    for i, (_, r) in enumerate(m.head(n_rotulos).iterrows()):
        ax.annotate(r['Feature'], (r['SHAP_medio_abs'], r['PSI']), xytext=(6, 6 if i % 2 == 0 else -13),
                    textcoords='offset points',
                    fontsize=9, fontweight='bold', color=PALETA_AZUL['texto'])
    ax.text(1.0, 0.25, ' Crítico (PSI > 0.25)', transform=ax.get_yaxis_transform(), va='bottom', ha='right',
            fontsize=9, color=PALETA_AZUL['acento'], fontweight='bold')
    ax.text(1.0, 0.10, ' Alerta (PSI > 0.10)', transform=ax.get_yaxis_transform(), va='bottom', ha='right',
            fontsize=9, color=PALETA_AZUL['principal'], fontweight='bold')
    ax.set(xlabel='Importância no modelo (|SHAP| médio)', ylabel='PSI (In-Time → OOT)',
           ylim=(0, max(0.3, m['PSI'].max() * 1.15)))
    _titulo_ax(ax, 'Matriz Drift × Importância das Features',
               'Tamanho do ponto = % do impacto SHAP · rótulos = maior risco (PSI × importância)')
    plt.tight_layout()
    _salvar_fig(fig, filepath)
    return m


def plot_comparativo_psi_features(X_ref: pd.DataFrame, X_atual: pd.DataFrame, features: List[str], bins: int = 10,
                                  rotulo_atual: str = 'OOT',
                                  figsize: Tuple[float, float] = (16, 4.8), filepath: Optional[Union[str, Path]] = None) -> None:
    """Distribuição In-Time x janela atual das features selecionadas, com PSI e status no título (portada do Projeto_06)."""
    fig, axes = plt.subplots(1, len(features), figsize=figsize)
    axes = np.atleast_1d(axes)
    for ax, feat in zip(axes, features):
        a, b = X_ref[feat].astype('float64').dropna(), X_atual[feat].astype('float64').dropna()
        psi = calcular_psi(a, b, bins=bins)
        lo = min(np.percentile(a, 1), np.percentile(b, 1))
        hi = max(np.percentile(a, 99), np.percentile(b, 99))
        a_, b_ = a[(a >= lo) & (a <= hi)], b[(b >= lo) & (b <= hi)]
        discreta = pd.concat([a_, b_]).nunique() <= 25
        if discreta:
            vals = np.sort(pd.concat([a_, b_]).unique())
            pa = a_.value_counts(normalize=True).reindex(vals, fill_value=0)
            pb = b_.value_counts(normalize=True).reindex(vals, fill_value=0)
            xi = np.arange(len(vals))
            ax.bar(xi - 0.2, pa.values, width=0.4, color=COR_IN_TIME, label='In-Time')
            ax.bar(xi + 0.2, pb.values, width=0.4, color=COR_OOT, label=rotulo_atual)
            passo = max(1, len(vals) // 8)
            ax.set_xticks(xi[::passo])
            ax.set_xticklabels([f'{v:g}' for v in vals[::passo]])
            ax.set_ylabel('Proporção' if ax is axes[0] else '')
        else:
            sns.kdeplot(a_.sample(min(len(a_), 60000), random_state=0), ax=ax, color=COR_IN_TIME, fill=True, alpha=0.30, lw=2.2, label='In-Time')
            sns.kdeplot(b_.sample(min(len(b_), 60000), random_state=0), ax=ax, color=COR_OOT, fill=True, alpha=0.20, lw=2.4, label=rotulo_atual)
            ax.set_ylabel('Densidade' if ax is axes[0] else '')
        status = _status_psi(psi)
        cor = PALETA_AZUL['acento'] if status == 'Crítico' else (PALETA_AZUL['principal'] if status == 'Alerta' else PALETA_AZUL['texto'])
        ax.set_title(f'{feat}\nPSI = {psi:.4f} · {status}', loc='left', fontsize=11.5, color=cor)
        ax.set_xlabel('Valor da feature (p1–p99)')
        ax.legend(loc='upper right', fontsize=9)
    _titulo_fig(fig, 'Feature Drift — Variáveis com Maior Risco de Deslocamento')
    plt.tight_layout()
    _salvar_fig(fig, filepath)


def plot_gini_temporal(df_temp: pd.DataFrame, col_bloco: str = 'semana', rotulo_bloco: str = 'Semana',
                       piso_gini: Optional[float] = None, figsize: Tuple[float, float] = (15, 5.8),
                       filepath: Optional[Union[str, Path]] = None) -> Dict[str, float]:
    """
    Estabilidade do GINI por bloco temporal (Treino OOF → Validação → OOT), portada de
    `plot_estabilidade_gini_temporal`: partições sombreadas, médias ponderadas por janela e piso opcional.
    """
    d = df_temp.dropna(subset=['GINI'])
    fig, ax = plt.subplots(figsize=figsize)
    _sombrear_particoes(ax, df_temp, col_bloco)
    ax.plot(d[col_bloco], d['GINI'], color=COR_OOT, lw=2.6, marker='o', ms=6, markerfacecolor='white',
            markeredgewidth=2, zorder=3, label='GINI semanal')
    for _, r in d.iterrows():
        ax.text(r[col_bloco], r['GINI'] + 0.012, f"{r['GINI']:.2f}", ha='center', fontsize=7.8, color=PALETA_AZUL['texto_sec'])
    medias = _linhas_media_janela(ax, d, col_bloco, 'GINI', '{:.3f}')
    if piso_gini is not None:
        ax.axhline(piso_gini, color=PALETA_AZUL['acento'], lw=1.3, ls=':', zorder=2)
        ax.annotate(f'piso de aceitação ({piso_gini:.2f})', xy=(d[col_bloco].min() - 0.4, piso_gini), xytext=(2, -9),
                    textcoords='offset points', va='top', ha='left', fontsize=9, color=PALETA_AZUL['acento'],
                    fontweight='bold', bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='none',
                                                 alpha=0.85), zorder=5)
    lo = np.nanmin([d['GINI'].min(), piso_gini if piso_gini is not None else np.inf])
    ax.set_ylim(max(0, lo - 0.10), min(1.02, d['GINI'].max() + 0.12))
    ax.set_xlim(df_temp[col_bloco].min() - 0.6, df_temp[col_bloco].max() + 2.6)
    ax.set_xlabel(f'{rotulo_bloco} (desde o início da base)')
    ax.set_ylabel('GINI (2·AUC − 1)')
    cv = d.groupby('Janela')['GINI'].agg(lambda s: s.std() / s.mean() * 100)
    _titulo_ax(ax, 'Estabilidade Temporal do GINI',
               ' | '.join(f'Coeficiente de variação {j}: {v:.1f}%' for j, v in cv.items()))
    plt.tight_layout()
    _salvar_fig(fig, filepath)
    return medias


# ------------------------------------------------------------------------------
# 6. MODELO x CLUSTERS (ARQUÉTIPOS DE FRAUDE DO 03_Segmentacao)
# ------------------------------------------------------------------------------
def obter_rotulos_clusters_fraude(
    df_master: pd.DataFrame, meta: Optional[Dict[str, Any]], mapa_arquetipos: Dict[int, str],
    caminho_cache: Optional[Union[str, Path]] = None, k: int = 10, n_neighbors: int = 40, min_dist: float = 0.0,
    random_state: int = 42, forcar_recalculo: bool = False, train_ratio: float = 0.60,
) -> pd.DataFrame:
    """
    Rótulos de cluster (K-Means K=10 no UMAP 10D) e arquétipo para as fraudes do `df_master`.

    O pré-processamento, o UMAP e o K-Means são ajustados SOMENTE nas fraudes do Treino
    (primeiros `train_ratio` cronológicos), reproduzindo EXATAMENTE a esteira do 03_Segmentacao
    (mesmo recorte, ordenação, pré-processamento, UMAP e semente). As fraudes de Validação e OOT
    são apenas projetadas (`aplicar_preparo_clusterizacao` -> `umap.transform` -> `kmeans.predict`),
    sem participar do ajuste. O resultado é salvo em cache (`caminho_cache`) para não recalcular o UMAP.
    """
    if caminho_cache is not None and Path(caminho_cache).exists() and not forcar_recalculo:
        df_cl = pd.read_parquet(caminho_cache)
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
    df_cl_treino = pd.DataFrame({'TransactionID': df_fraude['TransactionID'].values,
                                 'cluster': kmeans.labels_.astype(int), 'base_cluster': 'Treino (ajuste)'})

    # Fraudes fora do Treino (Validação + OOT): apenas projetadas no espaço ajustado
    mask_fora = (df_master['isFraud'] == 1) & ~df_master['TransactionID'].isin(df_fraude['TransactionID'])
    df_fora = df_master[mask_fora].sort_values('TransactionDT').reset_index(drop=True)
    partes = [df_cl_treino]
    if len(df_fora):
        X_fora = aplicar_preparo_clusterizacao(df_fora, preprocessador)
        rotulos_fora = kmeans.predict(modelo_umap.transform(X_fora))
        partes.append(pd.DataFrame({'TransactionID': df_fora['TransactionID'].values,
                                    'cluster': rotulos_fora.astype(int), 'base_cluster': 'Projetada (Val/OOT)'}))
    print(f"[CLUSTERS] Ajuste em {len(df_fraude):,} fraudes do Treino | projeção em {len(df_fora):,} fraudes de Validação/OOT")
    df_cl = pd.concat(partes, ignore_index=True)
    df_cl['arquetipo'] = df_cl['cluster'].map(mapa_arquetipos)
    if caminho_cache is not None:
        Path(caminho_cache).parent.mkdir(parents=True, exist_ok=True)
        df_cl.to_parquet(caminho_cache, index=False)
        print(f"[CLUSTERS] Rótulos salvos em cache: {caminho_cache}")
    return df_cl


def avaliar_modelo_por_grupo(df_fraudes: pd.DataFrame, col_grupo: str, col_janela: str = 'Janela',
                             ordem: Optional[List[Any]] = None) -> pd.DataFrame:
    """
    Desempenho do modelo DENTRO das fraudes de cada grupo (cluster/arquétipo) e janela:
    Recall, prejuízo total, evitado (VP) e vazado (FN) em US$, e score mediano.
    (Os clusters existem só para fraudes, então a métrica natural é a captura — Recall.)
    """
    linhas = []
    for (g, j), d in df_fraudes.groupby([col_grupo, col_janela], sort=False):
        barrada = d['bloquear'].values
        amt = d['amt_usd'].values
        linhas.append({
            'Grupo': g, 'Janela': j, 'Fraudes': len(d),
            'Recall_%': barrada.mean() * 100,
            'Prejuizo_Total_USD': amt.sum(),
            'Prejuizo_Evitado_USD': amt[barrada].sum(),
            'Prejuizo_Vazado_USD': amt[~barrada].sum(),
            '%_Vazado_USD': amt[~barrada].sum() / max(1e-9, amt.sum()) * 100,
            'Ticket_Medio_USD': amt.mean(),
            'Score_Mediano': float(np.median(d['prob_raw'])),
        })
    df = pd.DataFrame(linhas)
    tot_vaz = df.groupby('Janela')['Prejuizo_Vazado_USD'].transform('sum')
    df['%_do_Vazamento_da_Janela'] = df['Prejuizo_Vazado_USD'] / tot_vaz.replace(0, np.nan) * 100
    if ordem is not None:
        df['Grupo'] = pd.Categorical(df['Grupo'], categories=ordem, ordered=True)
    return df.sort_values(['Grupo', 'Janela']).reset_index(drop=True)


def plot_recall_grupos_janelas(df_grupo: pd.DataFrame, ordem: Optional[List[Any]] = None,
                               figsize: Tuple[float, float] = (16, 5.8), filepath: Optional[Union[str, Path]] = None) -> None:
    """
    (1) Recall por grupo In-Time x OOT; (2) prejuízo vazado (FN, US$) no OOT por grupo,
    com o pior grupo destacado no acento laranja.
    """
    ordem = ordem or list(dict.fromkeys(df_grupo['Grupo']))
    piv_r = df_grupo.pivot_table(index='Grupo', columns='Janela', values='Recall_%', observed=False).reindex(ordem)
    piv_v = df_grupo.pivot_table(index='Grupo', columns='Janela', values='Prejuizo_Vazado_USD', observed=False).reindex(ordem)
    piv_p = df_grupo.pivot_table(index='Grupo', columns='Janela', values='%_Vazado_USD', observed=False).reindex(ordem)
    x = np.arange(len(ordem))
    w = 0.38
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.bar(x - w / 2, piv_r.get('In-Time', pd.Series(np.nan, index=ordem)), width=w, color=COR_IN_TIME, label='In-Time')
    b = ax1.bar(x + w / 2, piv_r.get('OOT', pd.Series(np.nan, index=ordem)), width=w, color=COR_OOT, label='OOT')
    ax1.bar_label(b, fmt='%.1f%%', padding=3, fontsize=9, fontweight='bold', color=COR_OOT)
    ax1.set(xticks=x, ylim=(0, 105), ylabel='Recall (% das fraudes barradas)')
    ax1.set_xticklabels([str(o) for o in ordem], rotation=0 if len(ordem) <= 5 else 30, ha='center' if len(ordem) <= 5 else 'right')
    ax1.grid(axis='x', visible=False)
    ax1.legend(loc='upper right', ncol=2)
    _titulo_ax(ax1, 'Captura de Fraude (Recall) por Grupo', 'Queda do OOT vs. In-Time revela onde o modelo perde força')

    vaz = piv_v.get('OOT', pd.Series(0.0, index=ordem)).fillna(0) / 1e3
    i_pior = int(np.argmax(vaz.values))
    cores = [PALETA_AZUL['acento'] if i == i_pior else PALETA_AZUL['principal'] for i in range(len(ordem))]
    b2 = ax2.bar(x, vaz.values, color=cores, width=0.6)
    pct = piv_p.get('OOT', pd.Series(np.nan, index=ordem)).values
    for bar, v, p_ in zip(b2, vaz.values, pct):
        ax2.text(bar.get_x() + bar.get_width() / 2, v, f'US$ {v:,.1f}k\n({p_:.0f}% do grupo)', ha='center',
                 va='bottom', fontsize=8.8, fontweight='bold', color=PALETA_AZUL['texto'])
    ax2.set(xticks=x, ylim=(0, max(1e-6, vaz.max()) * 1.3), ylabel='Prejuízo vazado — FN (US$ mil)')
    ax2.set_xticklabels([str(o) for o in ordem], rotation=0 if len(ordem) <= 5 else 30, ha='center' if len(ordem) <= 5 else 'right')
    ax2.grid(axis='x', visible=False)
    _titulo_ax(ax2, 'Prejuízo que Escapou do Modelo no OOT', f'Maior vazamento: {ordem[i_pior]} (em laranja)')
    plt.tight_layout()
    _salvar_fig(fig, filepath)


def plot_score_fraudes_por_grupo(df_fraudes: pd.DataFrame, col_grupo: str, ordem: Optional[List[Any]] = None,
                                 faixa_corte: Optional[Tuple[float, float]] = None, col_score: str = 'prob_raw',
                                 figsize: Tuple[float, float] = (14, 5.5), filepath: Optional[Union[str, Path]] = None) -> None:
    """Boxplot do score das FRAUDES por grupo e janela: fraudes à esquerda da faixa de corte são as que passam."""
    ordem = ordem or list(dict.fromkeys(df_fraudes[col_grupo]))
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=df_fraudes, y=col_grupo, x=col_score, hue='Janela', order=ordem, hue_order=['In-Time', 'OOT'],
                palette={'In-Time': COR_IN_TIME, 'OOT': COR_OOT}, showfliers=False, width=0.62, linewidth=1.2, ax=ax,
                medianprops={'color': 'white', 'lw': 2})
    if faixa_corte is not None:
        ax.axvspan(*faixa_corte, color=PALETA_AZUL['acento'], alpha=0.14, lw=0, zorder=0)
        ax.text(np.mean(faixa_corte), 0.01, 'faixa de corte', transform=ax.get_xaxis_transform(), ha='center',
                va='bottom', fontsize=9, color=PALETA_AZUL['acento'], fontweight='bold')
    ax.set(xlim=(0, 1), xlabel='Score do modelo nas fraudes (probabilidade bruta)', ylabel='')
    ax.grid(axis='y', visible=False)
    ax.legend(title='', loc='lower right', bbox_to_anchor=(1.0, 1.01), ncol=2, frameon=False)
    _titulo_ax(ax, 'Score Atribuído às Fraudes por Grupo',
               'Quanto mais à esquerda da faixa de corte, mais "invisível" o grupo é para o modelo')
    plt.tight_layout()
    _salvar_fig(fig, filepath)


def plot_matriz_prioridade_clusters(df_cluster_oot: pd.DataFrame, rotulo_col: str = 'Grupo', cor_col: Optional[str] = None,
                                    paleta: Optional[Dict[str, str]] = None, figsize: Tuple[float, float] = (13, 6.5),
                                    filepath: Optional[Union[str, Path]] = None) -> None:
    """
    Matriz de prioridade dos clusters no OOT: X = Recall, Y = prejuízo vazado (US$), tamanho = nº de fraudes.
    Quadrante superior esquerdo (recall baixo + vazamento alto) = prioridade de ação.
    """
    d = df_cluster_oot.copy()
    fig, ax = plt.subplots(figsize=figsize)
    rec_med = np.average(d['Recall_%'], weights=d['Fraudes'])
    vaz_med = d['Prejuizo_Vazado_USD'].median() / 1e3
    ax.axvline(rec_med, color=PALETA_AZUL['cinza'], ls='--', lw=1)
    ax.axhline(vaz_med, color=PALETA_AZUL['cinza'], ls='--', lw=1)
    cores = d[cor_col].map(paleta).fillna(PALETA_AZUL['medio']) if (cor_col and paleta) else PALETA_AZUL['medio']
    ax.scatter(d['Recall_%'], d['Prejuizo_Vazado_USD'] / 1e3, s=60 + 1400 * d['Fraudes'] / d['Fraudes'].max(),
               c=cores, alpha=0.85, edgecolor='white', lw=1.5, zorder=3)
    for _, r in d.iterrows():
        ax.annotate(str(r[rotulo_col]), (r['Recall_%'], r['Prejuizo_Vazado_USD'] / 1e3), ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white', zorder=4)
    ax.text(0.01, 0.98, 'PRIORIDADE\nrecall baixo · vazamento alto', transform=ax.transAxes, va='top', fontsize=9.5,
            fontweight='bold', color=PALETA_AZUL['acento'])
    if cor_col and paleta:
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], marker='o', ls='', color=c, ms=9, label=g) for g, c in paleta.items() if g in set(d[cor_col])]
        ax.legend(handles=handles, loc='upper right', fontsize=9)
    ax.set(xlabel='Recall no OOT (% das fraudes do cluster barradas)', ylabel='Prejuízo vazado no OOT (US$ mil)')
    _titulo_ax(ax, 'Matriz de Prioridade dos Clusters de Fraude (OOT)',
               f'Tamanho = nº de fraudes · linhas = recall médio ({rec_med:.1f}%) e vazamento mediano (US$ {vaz_med:,.1f}k)')
    plt.tight_layout()
    _salvar_fig(fig, filepath)


def perfil_fn_vs_vp(df_fraudes: pd.DataFrame, X: pd.DataFrame, features: List[str], top_n: int = 12,
                    titulo: str = 'Raio-X das Fraudes que Passaram (FN) vs. Barradas (VP)',
                    figsize: Tuple[float, float] = (12, 6.5), filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Compara o perfil das fraudes aprovadas (FN) com o das barradas (VP) via diferença padronizada
    (Cohen's d = (média FN − média VP) / desvio combinado), robusta a sinais e escalas diferentes.
    """
    Xf = X.loc[df_fraudes.index, features].astype('float64')
    fn = ~df_fraudes['bloquear'].values
    a, b = Xf[fn], Xf[~fn]
    sd = np.sqrt((a.var() + b.var()) / 2).replace(0, np.nan)
    df = pd.DataFrame({'Média FN (passaram)': a.mean(), 'Média VP (barradas)': b.mean(),
                       'Diferença padronizada (d)': (a.mean() - b.mean()) / sd}).dropna()
    df = df.reindex(df['Diferença padronizada (d)'].abs().sort_values(ascending=False).index).head(top_n)
    plot = df.iloc[::-1]
    cores = [PALETA_AZUL['marinho'] if v > 0 else PALETA_AZUL['claro'] for v in plot['Diferença padronizada (d)']]
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(plot.index, plot['Diferença padronizada (d)'], color=cores, height=0.65)
    ax.bar_label(bars, fmt='%+.2f', padding=4, fontsize=9, fontweight='bold', color=PALETA_AZUL['texto'])
    ax.axvline(0, color=PALETA_AZUL['texto_sec'], lw=1)
    lim = plot['Diferença padronizada (d)'].abs().max() * 1.25
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Diferença padronizada (Cohen's d) · > 0: maior nas fraudes que PASSARAM")
    ax.grid(axis='y', visible=False)
    _titulo_ax(ax, titulo, f'{fn.sum():,} FN vs. {(~fn).sum():,} VP · |d| > 0.5 = diferença relevante')
    plt.tight_layout()
    _salvar_fig(fig, filepath)
    return df


# ------------------------------------------------------------------------------
# 7. INFERÊNCIA E ANÁLISE DO TEST (05_Solucao)
# ------------------------------------------------------------------------------
def aplicar_fe_ieee_cis_base(
    df_tr: pd.DataFrame,
    df_id: pd.DataFrame,
    n_ref: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Replica exatamente a engenharia de atributos determinística do 01_EDA.ipynb
    em tabelas brutas IEEE-CIS (ex.: test_transaction + test_identity).

    Normaliza nomes `id-XX` -> `id_XX` presentes em `test_identity.csv`, preserva
    os campos reais de negócio (`TransactionID`, `TransactionDT`, `amt_usd`, `ProductCD`)
    antes de qualquer downcast ou transformação `log1p`, e ajusta a escala de
    `freq_*` para o tamanho amostral de referência (`n_ref`, quando informado).
    """
    tr = df_tr.copy()
    id_df = df_id.rename(columns=lambda c: c.replace('id-', 'id_') if c.startswith('id-') else c).copy()

    contexto = {
        'TransactionID': tr['TransactionID'].values,
        'TransactionDT': tr['TransactionDT'].values.astype(float),
        'amt_usd': tr['TransactionAmt'].values.astype('float64'),
        'ProductCD': tr['ProductCD'].fillna('Missing').astype(str).values if 'ProductCD' in tr.columns else np.full(len(tr), 'Missing'),
    }

    # Cell 33 (01_EDA): temporais e centavos
    tr['tempo_hora'] = ((tr['TransactionDT'] // 3600) % 24).astype('int8')
    tr['tempo_dia_semana'] = ((tr['TransactionDT'] // (3600 * 24)) % 7).astype('int8')
    tr['tempo_dia_corrido'] = (tr['TransactionDT'] // (3600 * 24)).astype('int16')
    tr['valor_centavos'] = np.modf(tr['TransactionAmt'])[0].astype('float32')

    # Cell 35 (01_EDA): e-mail, nulos em transaction e flags de presença
    tr['email_mesmo_dominio'] = ((tr['P_emaildomain'] == tr['R_emaildomain']) & tr['P_emaildomain'].notnull()).astype('int8')
    tr['email_sufixo_P'] = tr['P_emaildomain'].str.split('.').str[-1]
    cols_tr_features = [c for c in tr.columns if c not in ['TransactionID', 'isFraud']]
    tr['cont_null_tr'] = tr[cols_tr_features].isnull().sum(axis=1).astype('int16')
    tr['flag_tem_dist2'] = tr['dist2'].notnull().astype('int8')
    tr['flag_tem_R_email'] = tr['R_emaildomain'].notnull().astype('int8')

    # Cell 37 (01_EDA): bloco M, cartão e anomalias V
    cols_m_bool = [c for c in ['M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9'] if c in tr.columns]
    tr['match_cont_T'] = (tr[cols_m_bool] == 'T').sum(axis=1).astype('int8')
    tr['match_cont_F'] = (tr[cols_m_bool] == 'F').sum(axis=1).astype('int8')
    tr['match_assinatura_M1M2M3'] = tr['M1'].fillna('X') + '_' + tr['M2'].fillna('X') + '_' + tr['M3'].fillna('X')
    tr['match_M4_cat'] = tr['M4'].fillna('Missing').astype('category')

    tr['cartao_tipo_cat'] = tr['card6'].replace({'debit or credit': 'debit', 'charge card': 'credit'}).fillna('missing')
    tr['cartao_bandeira_tipo'] = (tr['card4'].fillna('missing') + '_' + tr['cartao_tipo_cat']).astype('category')

    if 'V14' in tr.columns:
        tr['flag_V14_zero'] = (tr['V14'] == 0.0).astype('int8')
    cols_near_const = [c for c in ['V107', 'V117', 'V118', 'V119', 'V120', 'V121', 'V122'] if c in tr.columns]
    if cols_near_const:
        tr['flag_V_anomalia_quase_const'] = ((tr[cols_near_const] != 1.0) & tr[cols_near_const].notnull()).any(axis=1).astype('int8')
    if 'V305' in tr.columns:
        tr.drop(columns=['V305'], inplace=True)
    for col in cols_m_bool:
        tr[f'match_{col}_bin'] = tr[col].map({'T': 1, 'F': 0})

    # Cell 39 (01_EDA): renomeação C* -> vel_C* e D* -> delta_D*
    mapa_ren = {f'C{i}': f'vel_C{i}' for i in range(1, 15)}
    mapa_ren.update({f'D{i}': f'delta_D{i}' for i in range(1, 16)})
    tr.rename(columns={k: v for k, v in mapa_ren.items() if k in tr.columns}, inplace=True)

    # Cell 50 (01_EDA): nulos em identity + descarte esparso
    id_df['cont_null_id'] = id_df.isnull().sum(axis=1)
    if 'id_23' in id_df.columns:
        id_df['flag_checagem_rede_avancada'] = id_df['id_23'].notnull().astype('int8')
    id_df.drop(columns=['id_07', 'id_08', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27'],
               inplace=True, errors='ignore')

    # Cell 56 (01_EDA): features de segurança, histórico e dispositivo
    if 'id_34' in id_df.columns:
        id_df['seg_id34_score'] = id_df['id_34'].str.split(':').str[1].astype(float)
    flags_seg = [c for c in ['id_35', 'id_36', 'id_37', 'id_38'] if c in id_df.columns]
    if len(flags_seg) == 4:
        id_df['seg_assinatura_id35_38'] = (
            id_df['id_35'].astype(str) + '_' + id_df['id_36'].astype(str) + '_' +
            id_df['id_37'].astype(str) + '_' + id_df['id_38'].astype(str)
        )
        id_df['seg_cont_T_id35_38'] = (id_df[flags_seg] == 'T').sum(axis=1)
    id_df['hist_disp_ip_consistencia'] = id_df['id_28'].fillna('Miss') + '_' + id_df['id_29'].fillna('Miss')
    id_df['disp_tela_padrao'] = id_df['id_32'].map(
        lambda x: 1 if x in [24.0, 32.0] else (0 if pd.notnull(x) else -1)
    ).astype('int8')

    mapa_ordinal = {
        'id_12': ('hist_id12_bin', {'NotFound': 0, 'Found': 1}),
        'id_15': ('hist_id15_ord', {'Unknown': 0, 'New': 1, 'Found': 2}),
        'id_16': ('hist_id16_bin', {'NotFound': 0, 'Found': 1}),
        'id_28': ('hist_id28_bin', {'New': 0, 'Found': 1}),
        'id_29': ('hist_id29_bin', {'NotFound': 0, 'Found': 1}),
        'id_35': ('seg_id35_bin', {'F': 0, 'T': 1}),
        'id_36': ('seg_id36_bin', {'F': 0, 'T': 1}),
        'id_37': ('seg_id37_bin', {'F': 0, 'T': 1}),
        'id_38': ('seg_id38_bin', {'F': 0, 'T': 1}),
        'DeviceType': ('disp_tipo_bin', {'desktop': 0, 'mobile': 1}),
    }
    for col, (novo_nome, mapping) in mapa_ordinal.items():
        if col in id_df.columns:
            id_df[novo_nome] = id_df[col].map(mapping)

    if 'id_31' in id_df.columns:
        id_df['disp_navegador'] = id_df['id_31'].str.split(' ').str[0].str.lower().replace({
            'samsung': 'samsung', 'mobile': 'safari', 'google': 'chrome'
        })
    if 'id_33' in id_df.columns:
        id_df['disp_tela_largura'] = id_df['id_33'].str.split('x').str[0].astype(float)
        id_df['disp_tela_altura'] = id_df['id_33'].str.split('x').str[1].astype(float)
        id_df['disp_tela_area'] = id_df['disp_tela_largura'] * id_df['disp_tela_altura']
    if 'DeviceInfo' in id_df.columns:
        id_df['disp_marca'] = 'Other'
        for padrao, marca in [
            ('SM-|SAMSUNG', 'Samsung'), ('iOS|Mac|Apple', 'Apple'), ('Windows', 'Windows'),
            ('Moto|XT', 'Motorola'), ('LG-', 'LG'), ('Huawei|ALE-|HUAWEI', 'Huawei'),
        ]:
            id_df.loc[id_df['DeviceInfo'].str.contains(padrao, na=False, case=False), 'disp_marca'] = marca

    # Cell 61 (01_EDA): descarte textual, downcast e merge left
    id_df.drop(columns=['id_31', 'id_33', 'DeviceInfo'], inplace=True, errors='ignore')
    tr = safe_downcast(tr)
    id_df = safe_downcast(id_df)
    df_merged = tr.merge(id_df, on='TransactionID', how='left')

    # Cell 66 (01_EDA): razão card1 e frequency encoding (ajustado p/ escala amostral de referência `n_ref`)
    card1_mean_amt = df_merged.groupby('card1')['TransactionAmt'].transform('mean')
    df_merged['valor_razao_media_card1'] = (df_merged['TransactionAmt'] / (card1_mean_amt + 1e-5)).astype('float32')
    fator_escala = (float(n_ref) / max(len(df_merged), 1)) if n_ref is not None else 1.0
    for col in ['card1', 'card2', 'addr1', 'P_emaildomain']:
        if col in df_merged.columns:
            vc = df_merged[col].value_counts(dropna=False) * fator_escala
            df_merged[f'freq_{col}'] = df_merged[col].map(vc).astype('float32')

    return df_merged, contexto


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


def preparar_dados_teste_inferencia(
    df_test_tr: pd.DataFrame,
    df_test_id: pd.DataFrame,
    df_master: pd.DataFrame,
    features_modelo: List[str],
    meta: Optional[dict] = None,
    train_ratio: float = 0.60,
    test_ratio: float = 0.20,
) -> Dict[str, Any]:
    """
    Prepara o conjunto cego de teste (Kaggle) com paridade total em relação ao
    `01_EDA.ipynb` + `preparar_dados_modelagem`:
      1. Engenharia de atributos determinística (`aplicar_fe_ieee_cis_base`) com escala amostral `len(df_master)`
      2. Alinhamento `log1p` idêntico ao `df_master` (`_detectar_e_aplicar_log1p_referencia`)
      3. Alinhamento dos mapas `freq_*` com `df_master` (fallback escalado para categorias inéditas)
      4. Estatísticas `card1`/`*_freq` e `OrdinalEncoder` ajustados SOMENTE nos 60% de treino
         de `df_master`, registrando a taxa de categorias desconhecidas (`OOV = -1`).
    """
    from sklearn.preprocessing import OrdinalEncoder

    meta = meta or {}
    cols_log_meta = meta.get('log1p_features') or meta.get('cols_log1p')

    df_fe, contexto = aplicar_fe_ieee_cis_base(df_test_tr, df_test_id, n_ref=len(df_master))
    df_fe, cols_log = _detectar_e_aplicar_log1p_referencia(df_fe, df_master, cols_log_explicitas=cols_log_meta)

    # Alinha `freq_card1`, `freq_card2`, `freq_addr1`, `freq_P_emaildomain` ao mapa de referência do df_master
    # (mantendo o valor escalado e já em log1p/raw de `df_fe` como fallback para categorias inéditas no Test)
    for col in ['card1', 'card2', 'addr1', 'P_emaildomain']:
        f_col = f'freq_{col}'
        if col in df_fe.columns and col in df_master.columns and f_col in df_master.columns:
            mapa_ref = df_master.groupby(col, dropna=False)[f_col].first()
            mapeado = df_fe[col].map(mapa_ref)
            df_fe[f_col] = np.where(mapeado.notna(), mapeado, df_fe[f_col]).astype('float32')

    dados_prep = preparar_dados_modelagem(df_master=df_master, meta=meta, train_ratio=train_ratio, test_ratio=test_ratio)
    train_ref = df_master.sort_values('TransactionDT').iloc[:int(len(df_master) * train_ratio)]
    cat_cols = dados_prep['cat_cols']

    # Mesmo ajuste pós-split de `preparar_dados_modelagem` (razão card1, *_freq e OrdinalEncoder)
    card1_stats = train_ref.groupby('card1')['TransactionAmt'].mean()
    media_global = float(train_ref['TransactionAmt'].mean())
    df_fe['TransactionAmt_to_mean_card1'] = (
        df_fe['TransactionAmt'] / (df_fe['card1'].map(card1_stats).fillna(media_global) + 1e-5)
    ).astype('float32')
    for col in ['card1', 'card2', 'addr1', 'P_emaildomain']:
        if col in df_fe.columns and col in train_ref.columns:
            freq_map = train_ref[col].value_counts(dropna=False)
            df_fe[f'{col}_freq'] = df_fe[col].map(freq_map).fillna(0).astype('float32')

    registros_oov = []
    cats_ok = [c for c in cat_cols if c in df_fe.columns and c in train_ref.columns]
    if cats_ok:
        ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        ord_enc.fit(train_ref[cats_ok].astype(object).fillna('Missing').astype(str))
        enc_test = ord_enc.transform(df_fe[cats_ok].astype(object).fillna('Missing').astype(str)).astype('int32')
        df_fe[cats_ok] = enc_test
        for idx_c, c in enumerate(cats_ok):
            n_unk_test = int((enc_test[:, idx_c] == -1).sum())
            n_unk_val = int((dados_prep['X_test'][c] == -1).sum()) if c in dados_prep['X_test'].columns else 0
            n_unk_oot = int((dados_prep['X_oot'][c] == -1).sum()) if c in dados_prep['X_oot'].columns else 0
            registros_oov.append({
                'Feature': c,
                'No_Modelo_Final': c in features_modelo,
                'Categorias_Treino': int(len(ord_enc.categories_[idx_c])),
                'OOV_Val_%': n_unk_val / max(len(dados_prep['X_test']), 1) * 100,
                'OOV_OOT_%': n_unk_oot / max(len(dados_prep['X_oot']), 1) * 100,
                'OOV_Test_Qtd': n_unk_test,
                'OOV_Test_%': n_unk_test / max(len(df_fe), 1) * 100,
            })

    df_oov = pd.DataFrame(registros_oov).sort_values(
        ['No_Modelo_Final', 'OOV_Test_%'], ascending=[False, False]
    ).reset_index(drop=True) if registros_oov else pd.DataFrame()

    ausentes = [f for f in features_modelo if f not in df_fe.columns]
    if ausentes:
        raise ValueError(f"Features esperadas pelo modelo ausentes no Test após FE: {ausentes}")

    X_test_final = df_fe[features_modelo].replace([np.inf, -np.inf], np.nan).copy()
    X_in_time = pd.concat(
        [dados_prep['X_train'][features_modelo], dados_prep['X_test'][features_modelo]],
        ignore_index=True,
    )
    X_val = dados_prep['X_test'][features_modelo].copy()
    X_oot = dados_prep['X_oot'][features_modelo].copy()
    val_df = dados_prep['test_df'][['TransactionID', 'TransactionDT', 'TransactionAmt', 'isFraud']].copy()
    oot_df = dados_prep['oot_df'][['TransactionID', 'TransactionDT', 'TransactionAmt', 'isFraud']].copy()

    del df_fe, dados_prep, train_ref
    gc.collect()

    print(f"[Teste] Matriz pronta: {X_test_final.shape[0]:,} linhas × {X_test_final.shape[1]} features | "
          f"{len(cols_log)} colunas em escala log1p.")

    return {
        'X_test': X_test_final,
        'X_in_time': X_in_time,
        'X_val': X_val,
        'X_oot': X_oot,
        'val_df': val_df,
        'oot_df': oot_df,
        'cols_log1p': cols_log,
        'df_oov': df_oov,
        **contexto,
    }


def comparar_escoragem_particoes(
    df_val_scores: pd.DataFrame,
    df_oot_scores: pd.DataFrame,
    df_pred_test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Tabela comparativa de sanidade dos scores e acionamento operacional entre
    Validação (In-Time), OOT e Test (sem target).
    """
    particoes = [
        ('Validação (In-Time)', df_val_scores, 'bloquear'),
        ('Out-of-Time (OOT)', df_oot_scores, 'bloquear'),
        ('Test (Kaggle · Cego)', df_pred_test, 'pred_fraude'),
    ]
    linhas = []
    for nome, df_p, col_bloq in particoes:
        col_amt = 'amt_usd' if 'amt_usd' in df_p.columns else 'amt_real'
        bloq = df_p[col_bloq].astype(bool)
        amt = df_p[col_amt].astype('float64')
        prob_cal = df_p['prob_cal'].astype('float64')
        prob_raw = df_p['prob_raw'].astype('float64')
        th = df_p['threshold'].astype('float64') if 'threshold' in df_p.columns else df_p['threshold_dinamico'].astype('float64')
        gmv_total = float(amt.sum())
        gmv_bloq = float(amt[bloq].sum())
        linhas.append({
            'Partição': nome,
            'Transações': int(len(df_p)),
            'Score_Bruto_Médio_%': float(prob_raw.mean() * 100),
            'Prob_Cal_Média_%': float(prob_cal.mean() * 100),
            'Prob_Cal_P50_%': float(prob_cal.quantile(0.50) * 100),
            'Prob_Cal_P90_%': float(prob_cal.quantile(0.90) * 100),
            'Prob_Cal_P99_%': float(prob_cal.quantile(0.99) * 100),
            'Threshold_Médio_%': float(th.mean() * 100),
            'Taxa_Bloqueio_%': float(bloq.mean() * 100),
            'GMV_Retido_%': float(gmv_bloq / max(gmv_total, 1e-9) * 100),
        })
    return pd.DataFrame(linhas)


def relatorio_psi_vs_importancia_modelo(
    X_ref: pd.DataFrame,
    X_atual: pd.DataFrame,
    artefato: Dict[str, Any],
    bins: int = 10,
) -> pd.DataFrame:
    """
    Calcula o PSI e KS (In-Time -> Test) das features do modelo e cruza com a importância
    relativa no estimador final (`feature_importances_`), gerando `Risco_Drift = PSI * %_Importancia`.
    """
    features = list(artefato['features'])
    df_psi = relatorio_psi_features(X_ref, X_atual, features=features, bins=bins)
    df_psi = df_psi.rename(columns={'%_Nulos_OOT': '%_Nulos_Test'})

    imp = np.asarray(getattr(artefato['modelo'], 'feature_importances_', np.ones(len(features))), dtype='float64')
    imp_pct = imp / max(imp.sum(), 1e-9) * 100.0
    df_imp = pd.DataFrame({
        'Feature': features,
        'Splits_Modelo': imp.astype(int),
        '%_Importancia': imp_pct,
    })
    df_imp['Rank_Importancia'] = df_imp['Splits_Modelo'].rank(ascending=False, method='min').astype(int)

    df_out = df_psi.merge(df_imp, on='Feature', how='left')
    df_out['Risco_Drift'] = df_out['PSI'] * (df_out['%_Importancia'] / 100.0)
    return df_out.sort_values(['PSI', 'Risco_Drift'], ascending=[False, False]).reset_index(drop=True)


def resumir_predicoes_test(df_pred: pd.DataFrame, col_grupo: str) -> pd.DataFrame:
    """Agrega volumetria, taxa de acionamento/bloqueio, score calibrado médio e GMV retido por grupo."""
    df_tmp = df_pred.copy()
    if 'amt_bloqueado_usd' not in df_tmp.columns:
        df_tmp['amt_bloqueado_usd'] = np.where(df_tmp['pred_fraude'] == 1, df_tmp['amt_usd'], 0.0)
    g = (
        df_tmp.groupby(col_grupo, observed=False)
        .agg(
            Transacoes=('TransactionID', 'count'),
            Prev_Fraudes=('pred_fraude', 'sum'),
            Prob_Cal_Media=('prob_cal', 'mean'),
            Threshold_Medio=('threshold_dinamico', 'mean'),
            Ticket_Medio_USD=('amt_usd', 'mean'),
            Valor_Total_USD=('amt_usd', 'sum'),
            Valor_Bloqueado_USD=('amt_bloqueado_usd', 'sum'),
        )
        .reset_index()
    )
    g['%_Base'] = g['Transacoes'] / len(df_tmp) * 100
    g['Taxa_Prevista_%'] = g['Prev_Fraudes'] / g['Transacoes'] * 100
    g['Prob_Cal_Media_%'] = g['Prob_Cal_Media'] * 100
    g['Threshold_Medio_%'] = g['Threshold_Medio'] * 100
    total_bloq = max(float(df_tmp['amt_bloqueado_usd'].sum()), 1e-9)
    g['%_Exposicao_Bloq'] = g['Valor_Bloqueado_USD'] / total_bloq * 100
    return g


def plot_evolucao_predicoes_test(
    df_temp: pd.DataFrame,
    col_bloco: str = 'mes_test',
    rotulo_bloco: str = 'Mês do Test',
    taxa_ref_oot: Optional[float] = None,
    prob_ref_oot: Optional[float] = None,
    figsize: Tuple[float, float] = (14, 5.2),
    filepath: Optional[Union[str, Path]] = None,
) -> None:
    """
    Evolução temporal do acionamento no Test (sem ground-truth):
    volume em barras ao fundo, taxa de bloqueio (`pred_fraude = 1`), score calibrado médio (`prob_cal`)
    e linha de referência opcional da janela OOT.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax_vol = ax.twinx()
    ax_vol.bar(df_temp[col_bloco], df_temp['Transacoes'], width=0.62, color=PALETA_AZUL['cinza_claro'], alpha=0.55, zorder=0)
    ax_vol.set_ylim(0, df_temp['Transacoes'].max() * 3.1)
    ax_vol.set_yticks([])
    ax_vol.grid(False)
    for s in ax_vol.spines.values():
        s.set_visible(False)
    ax.set_zorder(ax_vol.get_zorder() + 1)
    ax.patch.set_visible(False)

    ax.plot(df_temp[col_bloco], df_temp['Taxa_Prevista_%'], color=COR_OOT, lw=2.6, marker='o', ms=6.5,
            markerfacecolor='white', markeredgewidth=2, label='Taxa de bloqueio no Test (pred_fraude = 1)', zorder=3)
    ax.plot(df_temp[col_bloco], df_temp['Prob_Cal_Media_%'], color=PALETA_AZUL['acento'], lw=2.0, ls='--',
            marker='s', ms=5, label='Score calibrado médio no Test (prob_cal)', zorder=3)

    if taxa_ref_oot is not None:
        ax.axhline(taxa_ref_oot, color=COR_IN_TIME, lw=1.6, ls=':', zorder=2,
                   label=f'Ref. taxa de bloqueio OOT ({taxa_ref_oot:.2f}%)')
    if prob_ref_oot is not None:
        ax.axhline(prob_ref_oot, color=PALETA_AZUL['principal'], lw=1.3, ls='-.', alpha=0.80, zorder=2,
                   label=f'Ref. prob_cal média OOT ({prob_ref_oot:.2f}%)')

    for _, r in df_temp.iterrows():
        ax.annotate(f"{r['Taxa_Prevista_%']:.1f}%", (r[col_bloco], r['Taxa_Prevista_%']),
                    xytext=(0, 8), textcoords='offset points', ha='center', fontsize=8.0,
                    fontweight='bold', color=COR_OOT)

    ymax_base = max(
        float(df_temp['Taxa_Prevista_%'].max()),
        float(df_temp['Prob_Cal_Media_%'].max()),
        float(taxa_ref_oot or 0.0),
    )
    ax.set(xlabel=f'{rotulo_bloco} · barras cinza = volume de transações', ylabel='Taxa / Score (%)', ylim=(0, ymax_base * 1.35))
    ax.legend(loc='upper left', ncol=2, frameon=True, facecolor='white', edgecolor='white', framealpha=0.9)
    taxa_global = df_temp['Prev_Fraudes'].sum() / df_temp['Transacoes'].sum() * 100
    prob_global = (df_temp['Prob_Cal_Media_%'] * df_temp['Transacoes']).sum() / df_temp['Transacoes'].sum()
    sub = f"Taxa de bloqueio global Test: {taxa_global:.2f}% | Score calibrado médio Test: {prob_global:.2f}%"
    if taxa_ref_oot is not None:
        sub += f" | Baseline OOT: {taxa_ref_oot:.2f}%"
    _titulo_ax(ax, f'Monitoramento Temporal de Acionamento no Test ({rotulo_bloco})', sub)
    plt.tight_layout()
    _salvar_fig(fig, filepath)


def plot_predicoes_por_categoria(
    df_resumo: pd.DataFrame, col_cat: str, titulo: str, subtitulo: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 4.8), filepath: Optional[Union[str, Path]] = None
) -> None:
    """Painel enxuto (2 gráficos) de taxa de bloqueio (%) e GMV retido (US$ mi) por segmento."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    x = np.arange(len(df_resumo))
    rotulos = df_resumo[col_cat].astype(str)

    bars1 = ax1.bar(x, df_resumo['Taxa_Prevista_%'], color=COR_OOT, width=0.58)
    ax1.bar_label(bars1, fmt='%.2f%%', padding=4, fontsize=9, fontweight='bold', color=PALETA_AZUL['texto'])
    ax1.set(xticks=x, xticklabels=rotulos, ylabel='Taxa de bloqueio (%)',
            ylim=(0, df_resumo['Taxa_Prevista_%'].max() * 1.25))
    ax1.grid(axis='x', visible=False)
    _titulo_ax(ax1, 'Taxa de Bloqueio Operacional (%)', 'Proporção com pred_fraude = 1 por segmento')

    vol_mi = df_resumo['Valor_Bloqueado_USD'] / 1e6
    bars2 = ax2.bar(x, vol_mi, color=PALETA_AZUL['principal'], width=0.58)
    ax2.bar_label(bars2, fmt='$%.2fM', padding=4, fontsize=9, fontweight='bold', color=PALETA_AZUL['texto'])
    ax2.set(xticks=x, xticklabels=rotulos, ylabel='GMV retido (US$ milhões)',
            ylim=(0, max(vol_mi.max(), 0.01) * 1.25))
    ax2.grid(axis='x', visible=False)
    _titulo_ax(ax2, 'GMV Retido sob Bloqueio (US$ mi)', 'Soma de amt_usd onde pred_fraude = 1')

    _titulo_fig(fig, titulo, subtitulo)
    plt.tight_layout()
    _salvar_fig(fig, filepath)


# ------------------------------------------------------------------------------
# 8. HIPÓTESE DE THRESHOLD POR GRUPAMENTO (03_Segmentacao — Seção 7.0)
# ------------------------------------------------------------------------------
def clusterizar_base_treino_e_projetar(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    X_oot: Optional[pd.DataFrame] = None,
    oot_df: Optional[pd.DataFrame] = None,
    features_cluster: Optional[List[str]] = None,
    k: int = 4,
    n_components_pca: int = 10,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Clusteriza 100% da base de Treino (`X_train`, legítimas + fraudes) sem data leakage
    usando `SimpleImputer(median) -> RobustScaler -> PCA(10D) -> KMeans(K)` ajustados
    EXCLUSIVAMENTE no Treino, e projeta os grupamentos na Validação (`X_val`).

    Ordena os IDs dos grupos (0 .. K-1) da maior para a menor taxa de fraude no Treino,
    garantindo numeração determinística e interpretável (`G1` = maior risco ... `GK` = menor risco).
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler

    cols = [c for c in (features_cluster or list(X_train.columns)) if c in X_train.columns]
    n_comp = min(n_components_pca, len(cols))

    pipe_cluster = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
        ('pca', PCA(n_components=n_comp, random_state=random_state)),
        ('kmeans', KMeans(n_clusters=k, random_state=random_state, n_init=10)),
    ])

    raw_train = pipe_cluster.fit_predict(X_train[cols])
    raw_val = pipe_cluster.predict(X_val[cols])
    raw_oot = pipe_cluster.predict(X_oot[cols]) if X_oot is not None else None

    # Reordena IDs (0..k-1) por taxa de fraude decrescente no Treino
    y_tr = np.asarray(train_df['isFraud'], dtype=float)
    taxas_raw = {cid: float(y_tr[raw_train == cid].mean()) if (raw_train == cid).any() else 0.0 for cid in range(k)}
    ordem_cids = sorted(range(k), key=lambda c: taxas_raw[c], reverse=True)
    re_map = {old_cid: new_cid for new_cid, old_cid in enumerate(ordem_cids)}

    g_train = np.vectorize(re_map.get)(raw_train)
    g_val = np.vectorize(re_map.get)(raw_val)
    g_oot = np.vectorize(re_map.get)(raw_oot) if raw_oot is not None else None

    amt_tr = obter_amt_real(train_df, 'TransactionAmt')
    prod_tr = train_df['ProductCD'].astype(str).values if 'ProductCD' in train_df.columns else np.full(len(train_df), 'W')

    linhas = []
    rotulos_map: Dict[int, str] = {}
    for gid in range(k):
        m_tr = (g_train == gid)
        m_va = (g_val == gid)
        prod_dom = pd.Series(prod_tr[m_tr]).value_counts().idxmax() if m_tr.any() else '-'
        tkt_med = float(amt_tr[m_tr].mean()) if m_tr.any() else 0.0
        tx_fr = float(y_tr[m_tr].mean() * 100) if m_tr.any() else 0.0
        rotulo = f"G{gid + 1} (Prod {prod_dom} · Fraude {tx_fr:.1f}% · ${tkt_med:.0f})"
        rotulos_map[gid] = rotulo
        item = {
            'Grupo_ID': gid,
            'Grupamento': rotulo,
            'Transacoes_Treino': int(m_tr.sum()),
            '%_Treino': float(m_tr.mean() * 100),
            '%_Validacao': float(m_va.mean() * 100),
            'Produto_Dominante': prod_dom,
            'Ticket_Medio_USD': tkt_med,
            'Ticket_Mediano_USD': float(np.median(amt_tr[m_tr])) if m_tr.any() else 0.0,
            'Taxa_Fraude_Treino_%': tx_fr,
        }
        if g_oot is not None:
            item['%_OOT'] = float((g_oot == gid).mean() * 100)
        linhas.append(item)

    df_perfil = pd.DataFrame(linhas)
    out = {
        'k': k,
        'grupos_train': g_train,
        'grupos_val': g_val,
        'rotulos_map': rotulos_map,
        'df_perfil': df_perfil,
        'pipeline_cluster': pipe_cluster,
    }
    if g_oot is not None:
        out['grupos_oot'] = g_oot
    return out


def otimizar_thresholds_por_grupamento(
    y_train,
    prob_cal_train,
    amt_train,
    grupos_train,
    rotulos_map: Optional[Dict[int, str]] = None,
    threshold_fixo_global: float = 0.0390,
    n_trials_optuna: int = 250,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Determina o melhor threshold fixo `t_g` para cada grupamento `g` no OOF do Treino:
      1. Busca Exata por grupamento (`otimizar_threshold_fixo` em cada grupo disjunto)
      2. Otimização conjunta via Optuna (`TPESampler`) maximizando o Lucro Líquido Total no OOF.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    y = np.asarray(y_train, dtype=int)
    p = np.asarray(prob_cal_train, dtype=float)
    a = np.asarray(amt_train, dtype=float)
    g = np.asarray(grupos_train, dtype=int)
    grupos_unicos = sorted(np.unique(g).tolist())
    rotulos_map = rotulos_map or {gid: f'G{gid + 1}' for gid in grupos_unicos}

    # 1. Busca exata vetorizada dentro de cada grupo + curvas de lucro para visualização
    mapa_th_exato: Dict[int, float] = {}
    curvas_por_grupo: Dict[str, pd.DataFrame] = {}
    grade_plot = np.linspace(0.005, 0.20, 300)

    for gid in grupos_unicos:
        mask = (g == gid)
        if mask.sum() > 0 and y[mask].sum() > 0:
            res_exato = otimizar_threshold_fixo(y[mask], p[mask], a[mask], t_max=0.35)
            th_g = float(res_exato['Threshold'])
        else:
            th_g = float(threshold_fixo_global)
        mapa_th_exato[gid] = th_g
        curvas_por_grupo[rotulos_map[gid]] = curva_lucro_threshold(
            y[mask], p[mask], a[mask], np.union1d(grade_plot, [th_g, threshold_fixo_global])
        )

    # 2. Otimização conjunta via Optuna (TPESampler)
    ganho_bloq_unit = np.where(y == 0, -0.04 * a, a)  # ganho marginal de bloquear vs. aprovar
    base_aprovar_tudo = float(np.where(y == 0, 0.02 * a, -a).sum())

    def _objective(trial):
        th_vec = np.empty(len(p), dtype=float)
        for gid in grupos_unicos:
            t_g = trial.suggest_float(f't_g{gid}', 0.005, 0.25)
            th_vec[g == gid] = t_g
        bloq = (p >= th_vec)
        return base_aprovar_tudo + float(ganho_bloq_unit[bloq].sum())

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=random_state))
    # Enfileira referências iniciais (corte fixo global e solução analítica por grupo)
    study.enqueue_trial({f't_g{gid}': float(np.clip(threshold_fixo_global, 0.005, 0.25)) for gid in grupos_unicos})
    study.enqueue_trial({f't_g{gid}': float(np.clip(mapa_th_exato[gid], 0.005, 0.25)) for gid in grupos_unicos})
    study.optimize(_objective, n_trials=n_trials_optuna, show_progress_bar=False)

    mapa_th_optuna = {gid: float(study.best_params[f't_g{gid}']) for gid in grupos_unicos}

    # 3. Tabela consolidada por grupamento no Treino (OOF)
    linhas = []
    for gid in grupos_unicos:
        mask = (g == gid)
        th_opt = mapa_th_optuna[gid]
        th_ex = mapa_th_exato[gid]
        r_fixo = calcular_score_financeiro(y[mask], p[mask] >= threshold_fixo_global, a[mask])
        r_opt = calcular_score_financeiro(y[mask], p[mask] >= th_opt, a[mask])
        linhas.append({
            'Grupo_ID': gid,
            'Grupamento': rotulos_map[gid],
            'Transacoes_Treino': int(mask.sum()),
            'Taxa_Fraude_%': float(y[mask].mean() * 100),
            'Ticket_Medio_USD': float(a[mask].mean()),
            'Prob_Cal_Media_%': float(p[mask].mean() * 100),
            'Threshold_Fixo_Global_%': float(threshold_fixo_global * 100),
            'Threshold_Optuna_%': float(th_opt * 100),
            'Threshold_Exato_%': float(th_ex * 100),
            'Delta_vs_Global_pp': float((th_opt - threshold_fixo_global) * 100),
            'Lucro_OOF_Fixo_Global_USD': float(r_fixo['Resultado_Liquido']),
            'Lucro_OOF_Grupamento_USD': float(r_opt['Resultado_Liquido']),
            'Ganho_OOF_vs_Global_USD': float(r_opt['Resultado_Liquido'] - r_fixo['Resultado_Liquido']),
            'Recall_%': float(r_opt['Recall'] * 100),
            'Precision_%': float(r_opt['Precision'] * 100),
            'Aprovacao_Legit_%': float(r_opt['Approval_Rate'] * 100),
        })

    return {
        'df_thresholds': pd.DataFrame(linhas),
        'mapa_th_optuna': mapa_th_optuna,
        'mapa_th_exato': mapa_th_exato,
        'curvas_por_grupo': curvas_por_grupo,
        'study': study,
    }


def otimizar_hibrido_grupamento_exponencial_optuna(
    y_train,
    prob_cal_train,
    amt_train,
    grupos_train,
    regra_dinamica: Dict[str, Any],
    n_trials_optuna: int = 200,
    random_state: int = 42,
) -> Dict[int, float]:
    """
    Otimiza via Optuna um fator multiplicativo `alpha_g` por grupamento sobre a curva
    exponencial por ticket `T_g(amt) = alpha_g * T_exp(amt)` no OOF do Treino.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    y = np.asarray(y_train, dtype=int)
    p = np.asarray(prob_cal_train, dtype=float)
    a = np.asarray(amt_train, dtype=float)
    g = np.asarray(grupos_train, dtype=int)
    grupos_unicos = sorted(np.unique(g).tolist())

    th_base = threshold_dinamico(a, regra_dinamica['t_min'], regra_dinamica['t_max'], regra_dinamica['k'])
    ganho_bloq_unit = np.where(y == 0, -0.04 * a, a)
    base_aprovar_tudo = float(np.where(y == 0, 0.02 * a, -a).sum())

    def _obj_hib(trial):
        fator = np.ones(len(p), dtype=float)
        for gid in grupos_unicos:
            fator[g == gid] = trial.suggest_float(f'alpha_g{gid}', 0.65, 1.45)
        bloq = (p >= (th_base * fator))
        return base_aprovar_tudo + float(ganho_bloq_unit[bloq].sum())

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=random_state))
    study.enqueue_trial({f'alpha_g{gid}': 1.0 for gid in grupos_unicos})
    study.optimize(_obj_hib, n_trials=n_trials_optuna, show_progress_bar=False)
    return {gid: float(study.best_params[f'alpha_g{gid}']) for gid in grupos_unicos}


def otimizar_exponencial_por_grupamento(
    y_train,
    prob_cal_train,
    amt_train,
    grupos_train,
    rotulos_map: Optional[Dict[int, str]] = None,
    regra_dinamica_global: Optional[Dict[str, Any]] = None,
    n_trials_optuna: int = 250,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Otimiza via Optuna (TPESampler) uma curva de threshold dinâmico exponencial própria
    para CADA grupamento `g` no OOF do Treino:
        T_g(amt) = t_min_g + (t_max_g - t_min_g) * exp(-k_g * amt)
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    y = np.asarray(y_train, dtype=int)
    p = np.asarray(prob_cal_train, dtype=float)
    a = np.asarray(amt_train, dtype=float)
    g = np.asarray(grupos_train, dtype=int)
    grupos_unicos = sorted(np.unique(g).tolist())
    rotulos_map = rotulos_map or {gid: f'G{gid + 1}' for gid in grupos_unicos}

    t_min_glob = float((regra_dinamica_global or {}).get('t_min', 0.0381))
    t_max_glob = float((regra_dinamica_global or {}).get('t_max', 0.0509))
    k_glob = float((regra_dinamica_global or {}).get('k', 0.01953))

    mapa_params_exp: Dict[int, Dict[str, float]] = {}
    linhas = []

    for gid in grupos_unicos:
        mask = (g == gid)
        y_g, p_g, a_g = y[mask], p[mask], a[mask]
        th_glob_g = threshold_dinamico(a_g, t_min_glob, t_max_glob, k_glob)
        r_glob = calcular_score_financeiro(y_g, p_g >= th_glob_g, a_g)

        if len(y_g) > 0 and y_g.sum() > 0:
            ganho_bloq_g = np.where(y_g == 0, -0.04 * a_g, a_g)
            base_aprov_g = float(np.where(y_g == 0, 0.02 * a_g, -a_g).sum())

            def _obj_exp_g(trial, _p=p_g, _a=a_g, _gb=ganho_bloq_g, _ba=base_aprov_g):
                t_max = trial.suggest_float('t_max', 0.030, 0.300)
                t_min = trial.suggest_float('t_min', 0.005, max(0.006, t_max - 0.005))
                k = trial.suggest_float('k', 0.0005, 0.080, log=True)
                th = threshold_dinamico(_a, t_min, t_max, k)
                return _ba + float(_gb[_p >= th].sum())

            st_g = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=random_state + int(gid)),
            )
            st_g.enqueue_trial({
                't_max': float(np.clip(t_max_glob, 0.030, 0.300)),
                't_min': float(np.clip(t_min_glob, 0.005, max(0.006, t_max_glob - 0.005))),
                'k': float(np.clip(k_glob, 0.0005, 0.080)),
            })
            st_g.optimize(_obj_exp_g, n_trials=n_trials_optuna, show_progress_bar=False)
            best_p = {
                't_min': float(st_g.best_params['t_min']),
                't_max': float(st_g.best_params['t_max']),
                'k': float(st_g.best_params['k']),
            }
        else:
            best_p = {'t_min': t_min_glob, 't_max': t_max_glob, 'k': k_glob}

        mapa_params_exp[gid] = best_p
        th_opt_g = threshold_dinamico(a_g, best_p['t_min'], best_p['t_max'], best_p['k'])
        r_opt = calcular_score_financeiro(y_g, p_g >= th_opt_g, a_g)

        linhas.append({
            'Grupo_ID': gid,
            'Grupamento': rotulos_map[gid],
            'Transacoes_Treino': int(mask.sum()),
            'Taxa_Fraude_%': float(y_g.mean() * 100),
            'Ticket_Medio_USD': float(a_g.mean()),
            't_max_Grupo_%': float(best_p['t_max'] * 100),
            't_min_Grupo_%': float(best_p['t_min'] * 100),
            'k_Grupo': float(best_p['k']),
            'Threshold_Medio_Grupo_%': float(th_opt_g.mean() * 100),
            'Lucro_OOF_Exp_Global_USD': float(r_glob['Resultado_Liquido']),
            'Lucro_OOF_Exp_Grupo_USD': float(r_opt['Resultado_Liquido']),
            'Ganho_OOF_vs_Exp_Global_USD': float(r_opt['Resultado_Liquido'] - r_glob['Resultado_Liquido']),
            'Recall_%': float(r_opt['Recall'] * 100),
            'Precision_%': float(r_opt['Precision'] * 100),
            'Aprovacao_Legit_%': float(r_opt['Approval_Rate'] * 100),
        })

    return {
        'df_exp_grupos': pd.DataFrame(linhas),
        'mapa_params_exp': mapa_params_exp,
    }


def aplicar_threshold_exponencial_por_grupamento(
    amt,
    grupos,
    mapa_params_exp: Dict[int, Dict[str, float]],
    regra_dinamica_global: Dict[str, Any],
) -> np.ndarray:
    """Calcula vetorizadamente o threshold exponencial específico `T_g(amt)` de cada grupamento."""
    a = np.asarray(amt, dtype=float)
    g = np.asarray(grupos, dtype=int)
    th_out = threshold_dinamico(a, regra_dinamica_global['t_min'], regra_dinamica_global['t_max'], regra_dinamica_global['k'])
    for gid, p_exp in mapa_params_exp.items():
        m = (g == int(gid))
        if m.any():
            th_out[m] = threshold_dinamico(a[m], p_exp['t_min'], p_exp['t_max'], p_exp['k'])
    return th_out


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


def comparar_estrategias_threshold_grupamento(
    bases_avaliacao: Dict[str, Dict[str, Any]],
    threshold_fixo_global: float,
    regra_dinamica: Dict[str, Any],
    mapa_th_k4: Dict[int, float],
    mapa_th_k10: Optional[Dict[int, float]] = None,
    mapa_alpha_hibrido_k4: Optional[Dict[int, float]] = None,
    mapa_exp_k4: Optional[Dict[int, Dict[str, float]]] = None,
    mapa_exp_k10: Optional[Dict[int, Dict[str, float]]] = None,
) -> pd.DataFrame:
    """
    Compara financeiramente e operacionalmente as políticas de corte sobre o mesmo modelo (`LGBM_Otimizado`):
      - Sem Modelo
      - Fixo Global Ótimo
      - Fixo por Grupamento (K=4 e K=10 · Optuna)
      - Dinâmico Exponencial Global (Atual)
      - Dinâmico Exponencial por Grupamento (K=4 e K=10 · Optuna)
      - Híbrido Grupamento K=4 × Exponencial (Optuna) [se informado]
    """
    linhas = []
    for nome_base, d in bases_avaliacao.items():
        y = np.asarray(d['y'], dtype=int)
        p_cal = np.asarray(d['prob_cal'], dtype=float)
        amt = np.asarray(d['amt'], dtype=float)
        g4 = np.asarray(d['grupos_k4'], dtype=int)
        th_exp = threshold_dinamico(amt, regra_dinamica['t_min'], regra_dinamica['t_max'], regra_dinamica['k'])
        th_k4 = np.vectorize(lambda cid: mapa_th_k4.get(int(cid), threshold_fixo_global))(g4)

        politicas: Dict[str, np.ndarray] = {
            '1. Sem Modelo': np.zeros(len(p_cal), dtype=bool),
            '2. Fixo Global Ótimo': p_cal >= threshold_fixo_global,
            '3. Fixo por Grupamento (K=4 · Optuna)': p_cal >= th_k4,
        }
        if mapa_th_k10 is not None and 'grupos_k10' in d:
            g10 = np.asarray(d['grupos_k10'], dtype=int)
            th_k10 = np.vectorize(lambda cid: mapa_th_k10.get(int(cid), threshold_fixo_global))(g10)
            politicas['4. Fixo por Grupamento (K=10 · Optuna)'] = p_cal >= th_k10

        politicas['5. Dinâmico Exponencial Global (Atual)'] = p_cal >= th_exp

        if mapa_exp_k4 is not None:
            th_exp_k4 = aplicar_threshold_exponencial_por_grupamento(amt, g4, mapa_exp_k4, regra_dinamica)
            politicas['6. Dinâmico Exponencial por Grupamento (K=4)'] = p_cal >= th_exp_k4

        if mapa_exp_k10 is not None and 'grupos_k10' in d:
            g10 = np.asarray(d['grupos_k10'], dtype=int)
            th_exp_k10 = aplicar_threshold_exponencial_por_grupamento(amt, g10, mapa_exp_k10, regra_dinamica)
            politicas['7. Dinâmico Exponencial por Grupamento (K=10)'] = p_cal >= th_exp_k10

        if mapa_alpha_hibrido_k4 is not None:
            fator_h = np.vectorize(lambda cid: mapa_alpha_hibrido_k4.get(int(cid), 1.0))(g4)
            politicas['8. Híbrido Escalar K=4 × Exponencial'] = p_cal >= (th_exp * fator_h)

        lucro_exp = float(
            calcular_score_financeiro(y, politicas['5. Dinâmico Exponencial Global (Atual)'], amt)['Resultado_Liquido']
        )

        for pol, bloq in politicas.items():
            r = calcular_score_financeiro(y, bloq, amt)
            linhas.append({
                'Base': nome_base,
                'Política': pol,
                'Lucro Líquido (US$)': float(r['Resultado_Liquido']),
                'Δ vs. Exponencial Global (US$)': float(r['Resultado_Liquido'] - lucro_exp),
                'Receita VN (+2%)': float(r['Receita_VN']),
                'Perda FP (−2%)': float(r['Penalidade_FP']),
                'Chargeback FN (−100%)': float(r['Prejuizo_FN']),
                'Aceitação Geral (%)': float((1 - (r['Qtd_VP'] + r['Qtd_FP']) / max(1, len(bloq))) * 100),
                'Aprovação Legítimas (%)': float(r['Approval_Rate'] * 100),
                'Recall (%)': float(r['Recall'] * 100),
                'Precision (%)': float(r['Precision'] * 100),
            })

    return pd.DataFrame(linhas)


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
        sinal = f"({delta:+,.0f} US$)" if abs(delta) >= 1 else "(Referência)"
        ax1.text(
            bar.get_width() + 0.35,
            bar.get_y() + bar.get_height() / 2,
            f"$ {r['Lucro Líquido (US$)']/1e3:.2f}k  {sinal}",
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
            label=f"{nome_curto} ($ {r['Lucro Líquido (US$)']/1e3:.1f}k)",
        )
        if eh_atual:
            ax2.annotate(
                f"Atual ($ {r['Lucro Líquido (US$)']/1e3:.1f}k)",
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


# ==============================================================================
# MANIFESTO DE EXPORTAÇÃO PÚBLICA (API DE UTILS)
# ==============================================================================
__all__ = [
    'histograma_simples',
    'plot_target_distribution',
    'plot_painel_fraude_bivariado',
    'plot_evolucao_temporal_fraude',
    'analisar_tipos_colunas',
    'analisar_valores_nulos',
    'analisar_nulos_por_faixa',
    'analisar_valores_unicos',
    'analisar_baixa_cardinalidade',
    'safe_downcast',
    'analise_correlacao_pares',
    'analise_correlação_pares',
    'calcular_iv_dataset',
    'analisar_vif',
    'analisar_e_transformar_caudas_longas',
    'preparar_dados_modelagem',
    'avaliar_classificador',
    'avaliar_classificacao_oot',
    'plot_feature_importance',
    'objective_lgbm',
    'objective_xgb',
    'objective_cat',
    'objective_lgbm_grouped',
    'objective_xgb_grouped',
    'objective_cat_grouped',
    'obter_menores_features',
    'podar_features_por_shap',
    'extrair_metricas_avaliacao',
    'exportar_artefatos_modelo',
    'preparar_dados_clusterizacao',
    'aplicar_preparo_clusterizacao',
    'recortar_treino_temporal',
    'projetar_pca',
    'plot_pca_variancia_acumulada',
    'projetar_umap',
    'curva_elbow',
    'avaliar_kmeans_range',
    'plot_elbow_comparativo',
    'plot_metricas_validacao',
    'plot_k_distance_dbscan',
    'avaliar_grade_dbscan',
    'plot_metricas_dbscan',
    'avaliar_grade_hdbscan',
    'plot_metricas_hdbscan',
    'avaliar_rotulos_clusterizacao',
    'plot_distribuicao_clusters',
    'plot_clusters_2d_plotly',
    'gerar_perfil_fraude_clusters',
    'plot_violino_perfil_clusters',
    'otimizar_dbscan_optuna',
    'otimizar_hdbscan_optuna',
    'plot_optuna_clusterizacao',
    'consolidar_comparacao_clusterizacao',
    'plot_heatmap_zscore_clusters',
    'plot_matriz_confusao_final',
    'PALETA_ARQUETIPOS_PADRAO',
    'COR_RUIDO_CLUSTER',
    'reverter_escala_log1p',
    'gerar_tabela_arquetipos',
    'plot_volume_vs_prejuizo_arquetipos',
    'plot_violino_arquetipos',
    'resumir_dinamica_temporal_arquetipos',
    'plot_evolucao_temporal_arquetipos',
    'plot_arquetipos_2d_plotly',
    'verificar_disponibilidade_hdbscan',
    'avaliar_viabilidade_hdbscan',
    'selecionar_config_hdbscan',
    'ajustar_hdbscan',
    'comparar_hdbscan_vs_arquetipos',
    'plot_crosstab_hdbscan_arquetipos',
    'veredito_viabilidade_hdbscan',
    # Avaliação OOT (04_Avaliacao)
    'aplicar_estilo_azul',
    'gerar_scores_oof_treino',
    'escorar_particao',
    'adicionar_bloco_temporal',
    'tabela_metricas_janelas',
    'formatar_tabela_metricas',
    'plot_matrizes_confusao_janelas',
    'plot_roc_pr_janelas',
    'plot_kde_scores_janelas',
    'plot_cdf_ks_janelas',
    'gerar_tabela_decis_ks',
    'plot_decis_janelas',
    'calcular_shap_lgbm',
    'ranking_shap',
    'plot_importancia_shap',
    'plot_shap_beeswarm',
    'plot_shap_janelas',
    'tabela_temporal',
    'plot_target_drift_temporal',
    'teste_qui_quadrado_temporal',
    'calcular_psi_scores',
    'plot_psi_score_faixas',
    'relatorio_psi_features',
    'plot_psi_vs_shap',
    'plot_comparativo_psi_features',
    'plot_gini_temporal',
    'obter_rotulos_clusters_fraude',
    'avaliar_modelo_por_grupo',
    'plot_recall_grupos_janelas',
    'plot_score_fraudes_por_grupo',
    'plot_matriz_prioridade_clusters',
    'perfil_fn_vs_vp',
    'PALETA_AZUL',
    'COR_IN_TIME',
    'COR_OOT',
    'COR_LEGITIMA',
    'COR_FRAUDE',
    'CMAP_AZUL',
    # Inferência e análise do Test (05_Solucao)
    'aplicar_regra_dinamica',
    'aplicar_fe_ieee_cis_base',
    'preparar_dados_teste_inferencia',
    'comparar_escoragem_particoes',
    'relatorio_psi_vs_importancia_modelo',
    'resumir_predicoes_test',
    'plot_evolucao_predicoes_test',
    'plot_predicoes_por_categoria',
    # Hipótese de threshold por grupamento (03_Segmentacao — Seção 7.0)
    'clusterizar_base_treino_e_projetar',
    'otimizar_thresholds_por_grupamento',
    'otimizar_hibrido_grupamento_exponencial_optuna',
    'otimizar_exponencial_por_grupamento',
    'aplicar_threshold_exponencial_por_grupamento',
    'comparar_estrategias_threshold_grupamento',
    'plot_thresholds_por_grupamento',
    'plot_curvas_exponenciais_por_grupamento',
    'plot_benchmark_grupamento_vs_exponencial',
]

