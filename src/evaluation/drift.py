"""Drift: PSI (score e features), KS shift, Qui-quadrado e Risco_Drift (PSI x importância).

Código migrado de `utils.py` (fachada de compatibilidade dos notebooks).
"""
from __future__ import annotations

from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import ks_2samp


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


def _status_psi(psi: float) -> str:
    if psi < 0.10:
        return 'Estável'
    if psi < 0.25:
        return 'Alerta'
    return 'Crítico'


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
