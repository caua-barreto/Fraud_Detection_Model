"""Calibração de probabilidades (isotônica OOF) e escoragem Out-of-Fold do Treino.

Código migrado de `utils.py` (fachada de compatibilidade dos notebooks).
"""
from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def gerar_probs_calibradas(classe_modelo, params: dict, X_train, y_train, X_val, groups, n_splits: int = 4,
                           metodo: str = 'isotonic'):
    """
    1. Gera probabilidades Out-of-Fold no Treino via GroupKFold temporal (sem leakage);
    2. Treina o modelo em 100% do Treino e prediz a Validação;
    3. Ajusta a Calibração Isotônica APENAS no OOF e aplica no OOF e na Validação.

    `metodo='sigmoid'` (opcional, via configs.yaml) usa a calibração de Platt (`CalibradorPlatt`).
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

    calibrador = CalibradorPlatt() if metodo == 'sigmoid' else IsotonicRegression(out_of_bounds='clip')
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
    if hasattr(calibrador, 'inverso'):  # CalibradorPlatt
        return float(calibrador.inverso(threshold_cal))
    xs = np.asarray(calibrador.X_thresholds_, dtype=float)
    ys = np.asarray(calibrador.y_thresholds_, dtype=float)
    idx = int(np.argmax(ys >= threshold_cal))
    if ys[idx] < threshold_cal:
        return float(xs[-1] + 1e-12)
    if idx == 0 or ys[idx] == ys[idx - 1]:
        return float(xs[idx])
    frac = (threshold_cal - ys[idx - 1]) / (ys[idx] - ys[idx - 1])
    return float(xs[idx - 1] + frac * (xs[idx] - xs[idx - 1]))


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


class CalibradorPlatt:
    """
    Calibração de Platt (regressão logística sobre o logit do score bruto), com a mesma
    interface usada pelo artefato (`fit`, `transform`, `fit_transform`, `predict`).
    """

    def __init__(self, C: float = 1e6):
        self.C = C

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, prob_raw, y):
        from sklearn.linear_model import LogisticRegression
        self.lr_ = LogisticRegression(C=self.C, max_iter=1000).fit(self._logit(prob_raw), np.asarray(y).astype(int))
        self.a_ = float(self.lr_.coef_[0, 0])
        self.b_ = float(self.lr_.intercept_[0])
        return self

    def transform(self, prob_raw):
        return self.lr_.predict_proba(self._logit(prob_raw))[:, 1]

    predict = transform

    def fit_transform(self, prob_raw, y):
        return self.fit(prob_raw, y).transform(prob_raw)

    def inverso(self, prob_cal: float) -> float:
        """Score bruto equivalente a uma probabilidade calibrada (para gráficos na escala bruta)."""
        p = float(np.clip(prob_cal, 1e-7, 1 - 1e-7))
        z = (np.log(p / (1 - p)) - self.b_) / self.a_
        return float(1 / (1 + np.exp(-z)))
