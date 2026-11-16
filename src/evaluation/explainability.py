"""Explicabilidade: SHAP (TreeExplainer) e ranking de |SHAP| médio.

Código migrado de `utils.py` (fachada de compatibilidade dos notebooks).
"""
from __future__ import annotations

from typing import Tuple, Iterable

import numpy as np
import pandas as pd

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None


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
