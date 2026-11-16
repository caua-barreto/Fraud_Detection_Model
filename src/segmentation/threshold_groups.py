"""Hipótese de threshold por grupamento (03_Segmentacao — Seção 7.0).

Código migrado de `utils.py` (fachada de compatibilidade dos notebooks).
"""
from __future__ import annotations

from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.data.loader import (  # noqa: F401
    obter_amt_real,
)
from src.models.threshold import (  # noqa: F401
    calcular_score_financeiro,
    curva_lucro_threshold,
    otimizar_threshold_fixo,
    threshold_dinamico,
)


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
