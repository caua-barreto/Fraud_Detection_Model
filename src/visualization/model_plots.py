"""Gráficos e tabelas estilizadas da modelagem (02_Modelagem): importância, Optuna, lucro e políticas.

Código migrado de `utils.py` (fachada de compatibilidade dos notebooks).
"""
from __future__ import annotations

from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.visualization.theme import (  # noqa: F401
    _PALETA_POLITICAS,
    _cor_modelo,
    _estilo_limpo,
    _limpar_bordas,
)
from src.models.threshold import (  # noqa: F401
    threshold_dinamico,
)
from src.visualization.theme import _mostrar_figura


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
    _mostrar_figura()

    return df_imp.sort_values('Importância', ascending=False).reset_index(drop=True).head(top_n)


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
    _mostrar_figura()

    resumo = pd.DataFrame({
        'Modelo': nomes,
        'Melhor_AUC': [v.max() for v in dados_box],
        'Mediana_AUC': [np.median(v) for v in dados_box],
        'Desvio_AUC': [v.std() for v in dados_box],
        'Trials': [len(v) for v in dados_box],
    }).sort_values('Melhor_AUC', ascending=False).reset_index(drop=True)
    return resumo


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
        _mostrar_figura()
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
    _mostrar_figura()


_ROTULO_CURTO_POLITICA = {
    'Sem Modelo': 'sem modelo',
    'Clássico 0.50': '0.50',
    'Fixo Ótimo': 'fixo',
    'Dinâmico (Optuna)': 'dinâmico',
}


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
    _mostrar_figura()


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
    _mostrar_figura()


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
    _mostrar_figura()


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
    _mostrar_figura()


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
    _mostrar_figura()


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
    _mostrar_figura()

    # Tabela de Resumo Flat e Direta
    df_resumo = pd.DataFrame([
        {'Partição': 'Consolidado', 'Classe Real': 'Legítima (0)', 'Decisão': 'Aprovada (TN)', 'Transações': tn, '% da Classe': f"{tn/n_legit:.2%}"},
        {'Partição': 'Consolidado', 'Classe Real': 'Legítima (0)', 'Decisão': 'Bloqueada (FP)', 'Transações': fp, '% da Classe': f"{fp/n_legit:.2%}"},
        {'Partição': 'Consolidado', 'Classe Real': 'Fraude (1)',   'Decisão': 'Aprovada (FN)', 'Transações': fn, '% da Classe': f"{fn/n_fraude:.2%}"},
        {'Partição': 'Consolidado', 'Classe Real': 'Fraude (1)',   'Decisão': 'Bloqueada (VP)', 'Transações': tp, '% da Classe': f"{tp/n_fraude:.2%}"},
    ])
    
    return df_resumo.set_index(['Partição', 'Classe Real', 'Decisão'])
