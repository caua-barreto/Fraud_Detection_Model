"""Métricas de avaliação: ROC-AUC, GINI, KS, PR-AUC, matriz de confusão, lucro por janela/semana.

Código migrado de `utils.py` (fachada de compatibilidade dos notebooks).
"""
from __future__ import annotations

from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold

from src.visualization.theme import (  # noqa: F401
    PALETA_AZUL,
)
from src.data.loader import (  # noqa: F401
    obter_amt_real,
)
from src.models.threshold import (  # noqa: F401
    calcular_score_financeiro,
    threshold_dinamico,
)


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
