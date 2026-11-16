"""Regra financeira (+MDR / -chargeback), threshold fixo ótimo e curva dinâmica exponencial.

Código migrado de `utils.py` (fachada de compatibilidade dos notebooks).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional, Dict, Any, Iterator, Tuple

import numpy as np
import pandas as pd

from src.data.loader import (  # noqa: F401
    obter_amt_real,
)


# ==============================================================================
# POLÍTICA FINANCEIRA (MDR SOBRE LEGÍTIMAS E PERDA SOBRE FRAUDES APROVADAS)
# ------------------------------------------------------------------------------
# Padrão do projeto (usado pelos notebooks): VN = +2% | FP = -2% | FN = -100% | VP = 0.
# Os scripts (main.py / configs.yaml) podem alterar as taxas APENAS dentro do
# gerenciador de contexto `politica_financeira(...)`; fora dele vale sempre o padrão,
# então as análises dos notebooks não sofrem interferência do configs.yaml.
# ==============================================================================
POLITICA_FINANCEIRA_PADRAO: Dict[str, float] = {'mdr': 0.02, 'perda_fraude': 1.00}
_POLITICA_ATIVA: Dict[str, float] = dict(POLITICA_FINANCEIRA_PADRAO)


def taxas_financeiras() -> Tuple[float, float]:
    """Retorna (mdr, perda_fraude) vigentes."""
    return float(_POLITICA_ATIVA['mdr']), float(_POLITICA_ATIVA['perda_fraude'])


@contextmanager
def politica_financeira(mdr: float = 0.02, perda_fraude: float = 1.00) -> Iterator[Dict[str, float]]:
    """Aplica temporariamente outra política financeira (ex.: experimentos do configs.yaml)."""
    anterior = dict(_POLITICA_ATIVA)
    _POLITICA_ATIVA.update({'mdr': float(mdr), 'perda_fraude': float(perda_fraude)})
    try:
        yield dict(_POLITICA_ATIVA)
    finally:
        _POLITICA_ATIVA.clear()
        _POLITICA_ATIVA.update(anterior)


def calcular_score_financeiro(y_true, y_pred_block, amt, exibir_detalhes=False, titulo=""):
    """
    Score financeiro conforme regra de negócio (em US$ reais):
      VP = Fraude evitada (0 custo direto) | VN = +0.02 * Amt
      FP = -0.02 * Amt (receita perdida)  | FN = -1.00 * Amt (chargeback)
    """
    mdr, perda_fraude = taxas_financeiras()
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

    receita_vn  = mdr * vol_vn
    penal_fp    = mdr * vol_fp
    prejuizo_fn = perda_fraude * vol_fn
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

    mdr, perda_fraude = taxas_financeiras()
    v_aprovar = np.where(y_s == 0, mdr * a_s, -perda_fraude * a_s)    # VN = +2% | FN = -100%
    v_bloquear = np.where(y_s == 0, -mdr * a_s, 0.0)                  # FP = -2%  | VP = 0

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


def tabela_risco_inerente(particoes: Dict[str, pd.DataFrame], col_target: str = 'isFraud') -> pd.DataFrame:
    """
    Calcula a DRE (Demonstrativo de Resultado) teórica assumindo que a empresa
    aprovasse 100% ou bloqueasse 100% das transações, sem usar nenhum modelo.
    """
    mdr, perda_fraude = taxas_financeiras()
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
            'Receita Potencial (+2%)': vol_legit * mdr,
            'Prejuízo Inerente (-100%)': vol_fraud,
            'LUCRO: Aprovar 100%': (vol_legit * mdr) - perda_fraude * vol_fraud,
            'LUCRO: Bloquear 100%': -(vol_legit * mdr),
        })

    df_res = pd.DataFrame(linhas)

    # Linha de Total
    total = df_res.sum(numeric_only=True)
    total['Base'] = 'TOTAL'
    total['Taxa Fraude'] = (df_res['Taxa Fraude'] * df_res['Transações']).sum() / df_res['Transações'].sum()
    df_res = pd.concat([df_res, total.to_frame().T], ignore_index=True)

    return df_res


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

    mdr, perda_fraude = taxas_financeiras()
    receita_vn  = (mdr * amt[vn]).sum()
    penal_fp    = (mdr * amt[fp]).sum()
    prejuizo_fn = (perda_fraude * amt[fn]).sum()

    lucro_liquido = receita_vn - penal_fp - prejuizo_fn

    return {
        'Lucro': lucro_liquido,
        'Aprovacao': vn.sum() / max(1, (y_true == 0).sum()),
        'Recall': (bloqueado & (y_true == 1)).sum() / max(1, (y_true == 1).sum())
    }


def threshold_dinamico(amt, t_min, t_max, k):
    """Curva de threshold por ticket: decai de t_max (micro-compras) até t_min (alto ticket)."""
    return t_min + (t_max - t_min) * np.exp(-k * np.asarray(amt, dtype=float))


def objective_threshold_dinamico(trial, y_true, probs, amt, limites: Optional[Dict[str, Any]] = None):
    """
    Função objetivo Optuna: maximiza o lucro do threshold dinâmico no OOF calibrado.

    `limites` (opcional, usado pelo configs.yaml) = {'t_max': [lo, hi], 't_min': [lo, hi], 'k': [lo, hi]}.
    Sem `limites`, usa o espaço de busca original do 02_Modelagem. O teto de `t_min` é
    sempre limitado a `t_max - 0.01` (o piso precisa ficar abaixo do teto).
    """
    lim = limites or {}
    tmax_lo, tmax_hi = lim.get('t_max') or (0.05, 0.35)
    tmin_lo, tmin_hi = lim.get('t_min') or (0.005, None)
    k_lo, k_hi = lim.get('k') or (0.001, 0.05)
    t_max = trial.suggest_float('t_max', tmax_lo, tmax_hi)                        # Teto para tickets muito baixos
    teto_tmin = t_max - 0.01 if tmin_hi is None else min(tmin_hi, t_max - 0.01)
    t_min = trial.suggest_float('t_min', tmin_lo, max(teto_tmin, tmin_lo + 1e-6))  # Piso para tickets altos
    k     = trial.suggest_float('k', k_lo, k_hi, log=True)                         # Taxa de queda

    th = threshold_dinamico(amt, t_min, t_max, k)
    return calcular_matriz_lucro(y_true, probs, th, amt)['Lucro']


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


# ==============================================================================
# PIPELINE REPRODUTÍVEL (scripts/02_train.py): BUSCA DA REGRA DE DECISÃO
# ==============================================================================
def otimizar_threshold_dinamico(
    y_true, probs, amt, n_trials: int = 300, seed: int = 42, limites: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Optuna (TPE) sobre a curva exponencial no OOF calibrado (Seção 7.3.1 do 02). Retorna t_max, t_min, k, lucro_oof."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(lambda trial: objective_threshold_dinamico(trial, y_true, probs, amt, limites=limites),
                   n_trials=n_trials)
    return {**study.best_params, 'lucro_oof': float(study.best_value)}


def regra_threshold_fixo(threshold: float, lucro_oof: Optional[float] = None) -> Dict[str, float]:
    """Representa o corte fixo no formato da regra dinâmica (t_min = t_max = threshold, k = 0)."""
    d = {'t_min': float(threshold), 't_max': float(threshold), 'k': 0.0}
    if lucro_oof is not None:
        d['lucro_oof'] = float(lucro_oof)
    return d
