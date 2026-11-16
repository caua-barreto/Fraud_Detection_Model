"""Treino e tuning: objetivos Optuna (LGBM/XGB/Cat) com GroupKFold mensal e avaliação por CV.

Código migrado de `utils.py` (fachada de compatibilidade dos notebooks).
"""
from __future__ import annotations

from typing import Union, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, GroupKFold


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


# ==============================================================================
# PIPELINE REPRODUTÍVEL (scripts/02_train.py): ESTIMADORES, BASELINE E TUNING
# ==============================================================================
def classe_estimador(nome: str):
    """Classe do estimador (`lightgbm` | `xgboost` | `catboost`)."""
    nome = nome.lower()
    if nome == 'lightgbm':
        from lightgbm import LGBMClassifier
        return LGBMClassifier
    if nome == 'xgboost':
        from xgboost import XGBClassifier
        return XGBClassifier
    if nome == 'catboost':
        from catboost import CatBoostClassifier
        return CatBoostClassifier
    raise ValueError(f"Estimador desconhecido: {nome}")


def params_fixos_estimador(nome: str, taxa_desbalanceamento: float, random_state: int = 42) -> dict:
    """Parâmetros não tunados (pesos de classe, seed, paralelismo, verbosidade) — iguais ao 02_Modelagem."""
    nome = nome.lower()
    if nome == 'lightgbm':
        return {'scale_pos_weight': taxa_desbalanceamento, 'random_state': random_state, 'n_jobs': -1, 'verbosity': -1}
    if nome == 'xgboost':
        return {'scale_pos_weight': taxa_desbalanceamento, 'eval_metric': 'auc', 'tree_method': 'hist',
                'random_state': random_state, 'n_jobs': -1, 'verbosity': 0}
    if nome == 'catboost':
        return {'scale_pos_weight': taxa_desbalanceamento, 'eval_metric': 'AUC', 'random_seed': random_state,
                'verbose': False, 'thread_count': -1}
    raise ValueError(f"Estimador desconhecido: {nome}")


def params_baseline_estimador(nome: str, taxa_desbalanceamento: float) -> dict:
    """Hiperparâmetros do modelo base da Seção 2.1 do 02_Modelagem (usado no ranking SHAP e na poda)."""
    nome = nome.lower()
    if nome == 'lightgbm':
        return {'scale_pos_weight': taxa_desbalanceamento, 'n_estimators': 100, 'num_leaves': 31,
                'random_state': 10, 'n_jobs': -1, 'verbosity': -1}
    if nome == 'xgboost':
        return {'scale_pos_weight': taxa_desbalanceamento, 'eval_metric': 'auc', 'n_estimators': 100,
                'max_depth': 6, 'random_state': 10, 'tree_method': 'hist', 'n_jobs': -1}
    if nome == 'catboost':
        return {'scale_pos_weight': taxa_desbalanceamento, 'eval_metric': 'AUC', 'iterations': 100, 'depth': 6,
                'random_seed': 42, 'verbose': False, 'thread_count': -1}
    raise ValueError(f"Estimador desconhecido: {nome}")


def objective_estimador(nome: str):
    """Função objetivo Optuna correspondente ao estimador."""
    return {'lightgbm': objective_lgbm, 'xgboost': objective_xgb, 'catboost': objective_cat}[nome.lower()]


def otimizar_hiperparametros(
    nome_estimador: str,
    X_train: pd.DataFrame,
    y_train,
    groups,
    taxa_desbalanceamento: float,
    n_trials: int = 15,
    cv_splits: int = 3,
    seed: int = 42,
):
    """
    Busca bayesiana (TPE) maximizando o ROC-AUC médio no GroupKFold mensal (Seção 5.1 do 02).
    Retorna `(best_params, study)`.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = GroupKFold(n_splits=cv_splits)
    objetivo = objective_estimador(nome_estimador)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(
        lambda trial: objetivo(trial, X_train, y_train, cv, taxa_desbalanceamento=taxa_desbalanceamento, groups=groups),
        n_trials=n_trials,
    )
    print(f"[Optuna] {nome_estimador}: melhor ROC-AUC CV = {study.best_value:.4f} ({n_trials} trials, seed={seed})")
    return dict(study.best_params), study
