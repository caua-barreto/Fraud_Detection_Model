"""
02 | Treinamento (reprodução do 02_Modelagem.ipynb)

df_master -> Split temporal 60/20/20 (OOT isolado)
  -> Seleção de features (all | poda por SHAP | lista customizada)
  -> Hiperparâmetros (Optuna GroupKFold mensal | params_override)
  -> Calibração OOF (isotônica | Platt)
  -> Regra de decisão (threshold fixo ótimo | curva dinâmica exponencial por ticket)
  -> Modelo em 100% do Treino -> artefato .joblib + metadata/metrics JSON

Uso:
    python scripts/02_train.py [--set model.tuning.run_optuna=false] [--set decision_policy.mode=fixed]
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import json  # noqa: E402

from src.config import Config, executar_script, salvar_json  # noqa: E402

ETAPA = 'train'
_PREFIXO = {'lightgbm': 'LGBM', 'xgboost': 'XGB', 'catboost': 'CAT'}
_SUFIXO = {'otimizado': 'Otimizado', 'max': 'Max'}


def _nome_modelo(cfg: Config) -> str:
    est = cfg['model']['estimator']
    fs = cfg['model']['feature_selection']
    if fs['mode'] == 'shap_prune':
        sufixo = _SUFIXO[fs.get('select', 'otimizado')]
    else:
        sufixo = {'all': 'All', 'custom_list': 'Custom'}[fs['mode']]
    return f"{_PREFIXO[est]}_{sufixo}"


def selecionar_features(cfg: Config, dados: dict, Cls, dir_met: Path) -> list:
    """Seleção de features conforme `model.feature_selection.mode`."""
    import pandas as pd
    from src.features.selection import obter_menores_features, curva_poda_features_shap
    from src.models.train import params_baseline_estimador

    fs = cfg['model']['feature_selection']
    X_train, y_train = dados['X_train'], dados['y_train']
    if fs['mode'] == 'all':
        return list(dados['features_cols'])
    if fs['mode'] == 'custom_list':
        faltantes = [c for c in fs['custom_list'] if c not in X_train.columns]
        if faltantes:
            raise ValueError(f"custom_list contém features inexistentes após o split: {faltantes}")
        return list(fs['custom_list'])

    # shap_prune — Seções 2.1, 3.2 e 4.1 do 02_Modelagem
    est = cfg['model']['estimator']
    params_base = {**params_baseline_estimador(est, dados['taxa_desbalanceamento']),
                   **(cfg['model'].get('baseline_params') or {})}
    print(f"[train] Modelo base ({est}) para ranking |SHAP|: {params_base}")
    modelo_base = Cls(**params_base)
    modelo_base.fit(X_train, y_train)

    X_amostra = X_train.sample(n=min(int(fs['shap_sample_size']), len(X_train)), random_state=int(fs['shap_random_state']))
    df_rank = obter_menores_features(modelo_base, X_amostra, bottom_n=X_amostra.shape[1], exibir=False) \
        .rename(columns={'fraqueza': 'SHAP_medio_abs'})
    ranking = df_rank.sort_values('SHAP_medio_abs', ascending=False)['Feature'].tolist()
    df_rank.sort_values('SHAP_medio_abs', ascending=False).to_csv(dir_met / 'train_ranking_shap_base.csv', index=False)
    if len(ranking) <= int(fs['min_features']) + int(fs['prune_step']):
        print(f"[train] Apenas {len(ranking)} features (min_features={fs['min_features']}) -> poda ignorada, usando todas.")
        return ranking

    df_curva, df_resumo, f_max, f_otim = curva_poda_features_shap(
        modelo=modelo_base, ranking_features=ranking,
        X_train=X_train, y_train=y_train, X_test=dados['X_test'], y_test=dados['y_test'],
        metrica='GINI', passo=int(fs['prune_step']), min_features=int(fs['min_features']),
        tolerancia=float(fs['gini_tolerance']),
        titulo=f'Poda de Features por SHAP || {est}: GINI vs. Nº de Features', cor='#0096C7',
    )
    df_curva.to_csv(dir_met / 'train_curva_poda_shap.csv', index=False)
    df_resumo.to_csv(dir_met / 'train_resumo_poda_shap.csv', index=False)
    escolhidas = list(f_otim) if fs.get('select', 'otimizado') == 'otimizado' else list(f_max)
    print(f"[train] Poda SHAP: Max={len(f_max)} | Otimizado={len(f_otim)} -> usando '{fs.get('select')}' "
          f"({len(escolhidas)} features)")
    return escolhidas


def run(cfg: Config) -> dict:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import brier_score_loss

    from src.data.loader import preparar_dados_modelagem, obter_amt_real
    from src.models.train import classe_estimador, params_fixos_estimador, otimizar_hiperparametros
    from src.models.calibrate import gerar_probs_calibradas
    from src.models.threshold import (
        politica_financeira, otimizar_threshold_fixo, otimizar_threshold_dinamico, regra_threshold_fixo,
        threshold_equilibrio,
    )
    from src.models.artifacts import montar_artefato_final, aplicar_regra_dinamica, exportar_modelo_final
    from src.evaluation.metrics import avaliar_modelo_final

    m, d, dp = cfg['model'], cfg['data'], cfg['decision_policy']
    seed = int(cfg['experiment']['random_state'])
    dir_met, dir_art = cfg.saida('metrics'), cfg.saida('artifacts')

    df_master = pd.read_parquet(cfg.entrada('processed', 'df_master.parquet'))
    with open(cfg.entrada('processed', 'features_metadata.json'), 'r', encoding='utf-8') as f:
        meta = json.load(f)

    with politica_financeira(dp['mdr_rate'], dp['fraud_loss_rate']):
        # 1. Split temporal estrito (Seção 1.3)
        dados = preparar_dados_modelagem(df_master=df_master, meta=meta,
                                         train_ratio=d['train_ratio'], test_ratio=d['val_ratio'])
        del df_master
        y_train, y_val = dados['y_train'], dados['y_test']
        grupos_mes, taxa = dados['grupos_mes'], dados['taxa_desbalanceamento']

        est = m['estimator']
        Cls = classe_estimador(est)

        # 2. Seleção de features (Seções 3–4)
        feats = selecionar_features(cfg, dados, Cls, dir_met)
        X_train, X_val = dados['X_train'][feats], dados['X_test'][feats]

        # 3. Hiperparâmetros (Seção 5)
        if m['tuning']['run_optuna']:
            tuned, study = otimizar_hiperparametros(
                est, X_train, y_train, grupos_mes, taxa, n_trials=int(m['tuning']['n_trials']),
                cv_splits=int(m['cv_splits']), seed=int(m['tuning']['optuna_seed']),
            )
            pd.DataFrame([{'trial': t.number, 'roc_auc_cv': t.value, **t.params} for t in study.trials]) \
                .to_csv(dir_met / 'train_optuna_trials.csv', index=False)
        else:
            tuned = dict(m['tuning']['params_override'])
            print(f"[train] run_optuna=false -> params_override: {tuned}")
        best_params = {**tuned, **params_fixos_estimador(est, taxa, random_state=seed)}

        # 4. Calibração OOF (Seção 6.0)
        calib = gerar_probs_calibradas(
            classe_modelo=Cls, params=best_params, X_train=X_train, y_train=y_train, X_val=X_val,
            groups=grupos_mes, n_splits=int(cfg['calibration']['oof_splits']), metodo=cfg['calibration']['method'],
        )
        calibracao = {
            base: {'taxa_real': float(np.mean(y_)), 'media_bruta': float(raw.mean()), 'media_calibrada': float(cal.mean()),
                   'brier_bruto': float(brier_score_loss(y_, raw)), 'brier_calibrado': float(brier_score_loss(y_, cal))}
            for base, y_, raw, cal in [('OOF Treino', y_train, calib['oof_raw'], calib['oof_cal']),
                                       ('Validação', y_val, calib['val_raw'], calib['val_cal'])]
        }
        print(f"[train] Calibração ({cfg['calibration']['method']}): {calibracao}")

        # 5. Regra de decisão (Seção 7) — busca SOMENTE no OOF do Treino
        amt_train, amt_val = obter_amt_real(dados['train_df']), obter_amt_real(dados['test_df'])
        fixo = otimizar_threshold_fixo(y_train, calib['oof_cal'], amt_train, t_max=float(dp['fixed_t_max']))
        if dp['mode'] == 'dynamic_exp':
            params_regra = otimizar_threshold_dinamico(
                y_train, calib['oof_cal'], amt_train, n_trials=int(dp['optuna_trials_threshold']),
                seed=int(dp['optuna_seed']), limites=dp.get('bounds'),
            )
        else:
            params_regra = regra_threshold_fixo(fixo['Threshold'], fixo['Lucro'])
        print(f"[train] Corte fixo ótimo (OOF): {fixo['Threshold']:.4%} | regra '{dp['mode']}': {params_regra}")

        # 6. Modelo final em 100% do Treino (Seção 8.1) — `modelo_full` já é o ajuste em 100% do Treino
        nome_modelo = _nome_modelo(cfg)
        artefato = montar_artefato_final(
            nome_modelo=nome_modelo, modelo=calib['modelo_full'], calibrador=calib['calibrador'], features=feats,
            params_dinamico=params_regra, threshold_fixo_ref=fixo['Threshold'], n_amostras_treino=len(X_train),
        )
        if dp['mode'] == 'fixed':
            artefato['regra_decisao']['tipo'] = 'threshold_fixo'
            artefato['regra_decisao']['formula'] = 'bloquear se prob_calibrada >= t_min (t_min = t_max, k = 0)'
        artefato['regra_decisao']['politica_financeira'] = {
            'mdr': float(dp['mdr_rate']), 'perda_fraude': float(dp['fraud_loss_rate']),
            'threshold_equilibrio_teorico': threshold_equilibrio(dp['mdr_rate'], dp['fraud_loss_rate']),
        }
        artefato['experimento'] = cfg['experiment']['name']

        # 7. Avaliação Treino (Teste CV) x Validação (Seção 8.2) e exportação (Seção 8.3)
        inf_val = aplicar_regra_dinamica(artefato, X_val, amt_val)
        bases_final = {
            'Treino (Teste CV)': {'y': y_train, 'prob_raw': calib['oof_raw'], 'prob_cal': calib['oof_cal'],
                                  'amt': amt_train, 'groups': grupos_mes},
            'Validação': {'y': y_val, 'prob_raw': inf_val['prob_raw'], 'prob_cal': inf_val['prob_cal'], 'amt': amt_val},
        }
        df_metricas = avaliar_modelo_final(artefato, bases_final)

    caminhos = exportar_modelo_final(artefato, df_metricas, dir_art)
    df_metricas.to_csv(dir_met / 'train_metricas_treino_vs_validacao.csv')
    salvar_json({
        'modelo': nome_modelo, 'estimador': est, 'n_features': len(feats), 'features': feats,
        'hiperparametros_tunados': tuned, 'optuna_executado': bool(m['tuning']['run_optuna']),
        'calibracao': calibracao, 'threshold_fixo_oof': fixo, 'regra_decisao': artefato['regra_decisao'],
    }, dir_met / 'train_resumo.json')
    return {k: str(v) for k, v in caminhos.items()}


if __name__ == '__main__':
    executar_script(ETAPA, __doc__, run)
