"""Gráficos da avaliação In-Time vs. OOT (04_Avaliacao) e do Test (05_Submissao).

Código migrado de `utils.py` (fachada de compatibilidade dos notebooks).
"""
from __future__ import annotations

from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None

from src.visualization.theme import (  # noqa: F401
    CMAP_AZUL,
    CMAP_AZUL_SHAP,
    COR_FRAUDE,
    COR_IN_TIME,
    COR_LEGITIMA,
    COR_OOT,
    PALETA_AZUL,
    _CORES_PARTICAO,
    _salvar_fig,
    _titulo_ax,
    _titulo_fig,
)
from src.evaluation.drift import (  # noqa: F401
    _status_psi,
    calcular_psi,
)


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
