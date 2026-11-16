"""Montagem, aplicação e exportação do artefato final (joblib + metadata/metrics JSON).

Código migrado de `utils.py` (fachada de compatibilidade dos notebooks).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

from src.models.threshold import (  # noqa: F401
    threshold_dinamico,
)
from src.evaluation.metrics import (  # noqa: F401
    extrair_metricas_avaliacao,
)


def exportar_artefatos_modelo(
    nome_modelo,
    modelo,
    diretorio_destino,
    threshold_otimo,
    features_utilizadas,
    X_train,
    y_train,
    X_test,
    y_test,
    X_oot=None,
    y_oot=None,
):
    """
    Exporta o modelo em .joblib e grava os arquivos estruturados
    metadata.json e metrics.json (incluindo OOT quando fornecido).
    """
    import json
    from pathlib import Path
    try:
        import joblib
    except ImportError:
        print("[AVISO] 'joblib' não está instalado. Execute: pip install joblib")
        return

    dest = Path(diretorio_destino)
    dest.mkdir(parents=True, exist_ok=True)

    # 1. Serialização do estimador
    arquivo_modelo = dest / f"{nome_modelo}.joblib"
    joblib.dump(modelo, arquivo_modelo)

    # 2. Extração dos hiperparâmetros
    hiperparametros = {}
    if hasattr(modelo, 'get_params'):
        try:
            params_raw = modelo.get_params(deep=False)
            for k, v in params_raw.items():
                if isinstance(v, (np.integer,)):
                    hiperparametros[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    hiperparametros[k] = float(v)
                elif isinstance(v, (int, float, str, bool, list, dict)) or v is None:
                    hiperparametros[k] = v
                else:
                    hiperparametros[k] = str(v)
        except Exception:
            hiperparametros = {'info': 'Nao foi possivel serializar get_params()'}

    # 3. Geração do metadata.json
    metadata = {
        'nome_modelo': nome_modelo,
        'classe': f"{modelo.__class__.__module__}.{modelo.__class__.__name__}",
        'hiperparametros': hiperparametros,
        'features_utilizadas': list(features_utilizadas),
        'total_features': len(features_utilizadas),
        'tamanho_historico_treino': {
            'linhas': int(len(X_train)),
            'taxa_alvo': float(np.mean(y_train))
        },
        'tamanho_historico_teste': {
            'linhas': int(len(X_test)),
            'taxa_alvo': float(np.mean(y_test))
        }
    }
    if X_oot is not None and y_oot is not None:
        metadata['tamanho_historico_oot'] = {
            'linhas': int(len(X_oot)),
            'taxa_alvo': float(np.mean(y_oot))
        }

    arquivo_metadata = dest / f"{nome_modelo}_metadata.json"
    with open(arquivo_metadata, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # 4. Geração do metrics.json
    metricas_treino = extrair_metricas_avaliacao(modelo, X_train, y_train, threshold=threshold_otimo)
    metricas_teste  = extrair_metricas_avaliacao(modelo, X_test, y_test, threshold=threshold_otimo)

    metrics_payload = {
        'nome_modelo': nome_modelo,
        'threshold_otimo': float(threshold_otimo),
        'metricas_treino': metricas_treino,
        'metricas_teste': metricas_teste
    }
    if X_oot is not None and y_oot is not None:
        metrics_payload['metricas_oot'] = extrair_metricas_avaliacao(modelo, X_oot, y_oot, threshold=threshold_otimo)

    arquivo_metrics = dest / f"{nome_modelo}_metrics.json"
    with open(arquivo_metrics, 'w', encoding='utf-8') as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    print(f"Artefatos de {nome_modelo} exportados com sucesso em {dest}:")
    print(f" - {arquivo_modelo.name}")
    print(f" - {arquivo_metadata.name}")
    print(f" - {arquivo_metrics.name}")


# ------------------------------------------------------------------------------
# MODELO FINAL (SEÇÃO 8): ARTEFATO, REGRA DINÂMICA, AVALIAÇÃO E EXPORTAÇÃO
# ------------------------------------------------------------------------------
def montar_artefato_final(
    nome_modelo: str, modelo, calibrador, features: List[str],
    params_dinamico: Dict[str, float], threshold_fixo_ref: Optional[float] = None,
    n_amostras_treino: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Empacota tudo o que a inferência precisa em um único objeto serializável:
    estimador (treinado em 100% do Treino), calibrador isotônico, lista de features e parâmetros da regra dinâmica.
    """
    return {
        'nome_modelo': nome_modelo,
        'modelo': modelo,
        'calibrador': calibrador,
        'features': list(features),
        'treinado_100pct_treino': True,
        'n_amostras_treino': None if n_amostras_treino is None else int(n_amostras_treino),
        'regra_decisao': {
            'tipo': 'threshold_dinamico',
            'formula': 'bloquear se prob_calibrada >= t_min + (t_max - t_min) * exp(-k * TransactionAmt_USD)',
            't_min': float(params_dinamico['t_min']),
            't_max': float(params_dinamico['t_max']),
            'k': float(params_dinamico['k']),
            'lucro_oof_optuna': None if 'lucro_oof' not in params_dinamico else float(params_dinamico['lucro_oof']),
            'threshold_fixo_referencia': None if threshold_fixo_ref is None else float(threshold_fixo_ref),
        },
    }


def aplicar_regra_dinamica(artefato: Dict[str, Any], X, amt_real) -> Dict[str, np.ndarray]:
    """
    Inferência com o artefato final: probabilidade bruta -> calibrada -> corte por valor -> decisão.
    `amt_real` é o valor da transação em US$ reais (não em log).
    """
    X_ = X[artefato['features']] if hasattr(X, 'columns') else X
    regra = artefato['regra_decisao']
    prob_raw = artefato['modelo'].predict_proba(X_)[:, 1]
    prob_cal = artefato['calibrador'].transform(prob_raw)
    th = threshold_dinamico(amt_real, regra['t_min'], regra['t_max'], regra['k'])
    return {'prob_raw': prob_raw, 'prob_cal': prob_cal, 'threshold': th, 'bloquear': prob_cal >= th}


def _serializar_params(modelo) -> Dict[str, Any]:
    """Hiperparâmetros do estimador em tipos nativos (JSON-safe)."""
    saida = {}
    for k, v in modelo.get_params(deep=False).items():
        if isinstance(v, np.integer):
            saida[k] = int(v)
        elif isinstance(v, np.floating):
            saida[k] = float(v)
        elif isinstance(v, (int, float, str, bool, list, dict)) or v is None:
            saida[k] = v
        else:
            saida[k] = str(v)
    return saida


def exportar_modelo_final(
    artefato: Dict[str, Any], df_metricas: pd.DataFrame, diretorio_destino, nome_arquivo: str = 'modelo_fraude_final',
) -> Dict[str, Path]:
    """
    Exporta o modelo final (treinado em 100% da base de Treino) e seus relatórios completos:
      - <nome>.joblib          : artefato completo (modelo 100% treino + calibrador + features + regra dinâmica);
      - <nome>_metadata.json   : classe, hiperparâmetros, features, parâmetros do threshold dinâmico e bases;
      - <nome>_metrics.json    : métricas estatísticas, operacionais do threshold dinâmico, financeiras e gap.
    """
    import joblib
    from datetime import datetime

    dest = Path(diretorio_destino)
    dest.mkdir(parents=True, exist_ok=True)
    modelo = artefato['modelo']
    regra = artefato['regra_decisao']

    caminhos = {
        'artefato': dest / f'{nome_arquivo}.joblib',
        'metadata': dest / f'{nome_arquivo}_metadata.json',
        'metrics': dest / f'{nome_arquivo}_metrics.json',
    }

    joblib.dump(artefato, caminhos['artefato'])

    cols_bases = [c for c in df_metricas.columns if not c.startswith('Gap')]
    cols_gap   = [c for c in df_metricas.columns if c.startswith('Gap')]

    metadata = {
        'nome_modelo': artefato['nome_modelo'],
        'data_exportacao': datetime.now().isoformat(timespec='seconds'),
        'treinado_100pct_treino': bool(artefato.get('treinado_100pct_treino', True)),
        'n_amostras_treino_100pct': artefato.get('n_amostras_treino'),
        'classe': f"{modelo.__class__.__module__}.{modelo.__class__.__name__}",
        'calibracao': f"{artefato['calibrador'].__class__.__name__} (ajustada no OOF do Treino)",
        'hiperparametros': _serializar_params(modelo),
        'regra_decisao': regra,
        'total_features': len(artefato['features']),
        'features_utilizadas': artefato['features'],
        'bases_avaliadas_modelagem': {
            b: {'linhas': int(df_metricas.loc['Transações', b]), 'taxa_fraude': float(df_metricas.loc['Taxa Fraude', b])}
            for b in cols_bases
        },
        'uso': "artefato = joblib.load(...); utils.aplicar_regra_dinamica(artefato, X, valor_transacao_usd)",
    }
    with open(caminhos['metadata'], 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    def _extrair_bloco(col):
        s = df_metricas[col]
        return {
            'independentes_threshold': {
                'ROC-AUC': float(s['ROC-AUC']),
                'GINI': float(s['GINI']),
                'PR-AUC': float(s['PR-AUC']),
                'KS': float(s['KS']),
            },
            'regra_threshold_dinamico': {
                'Precision': float(s['Precision']),
                'Recall': float(s['Recall']),
                'F1-Score': float(s['F1-Score']),
                'Aprovacao_Legitimas': float(s['Aprovação']),
                'Aceitacao_Geral': float(s['Aceitação Geral']),
                'Qtd_VP': None if pd.isna(s['Qtd_VP']) else int(s['Qtd_VP']),
                'Qtd_FP': None if pd.isna(s['Qtd_FP']) else int(s['Qtd_FP']),
                'Qtd_FN': None if pd.isna(s['Qtd_FN']) else int(s['Qtd_FN']),
                'Qtd_VN': None if pd.isna(s['Qtd_VN']) else int(s['Qtd_VN']),
            },
            'financeiro_usd': {
                'Receita_VN_USD': None if pd.isna(s['Receita VN']) else float(s['Receita VN']),
                'Perda_FP_USD': None if pd.isna(s['Perda FP']) else float(s['Perda FP']),
                'Chargeback_FN_USD': None if pd.isna(s['Chargeback FN']) else float(s['Chargeback FN']),
                'Lucro_Liquido_USD': None if pd.isna(s['Lucro Líquido']) else float(s['Lucro Líquido']),
                'Lucro_Sem_Modelo_USD': None if pd.isna(s['Lucro Sem Modelo']) else float(s['Lucro Sem Modelo']),
                'Lucro_por_Transacao_USD': float(s['Lucro por Transação']),
                'Margem_sobre_Volume': float(s['Margem s/ Volume']),
            },
        }

    payload_metrics = {
        'nome_modelo': artefato['nome_modelo'],
        'treinado_100pct_treino': bool(artefato.get('treinado_100pct_treino', True)),
        'threshold_dinamico_params': regra,
        'metricas_por_base': {b: _extrair_bloco(b) for b in cols_bases},
        'gap_validacao_vs_cv': {
            k: float(df_metricas.loc[k, cols_gap[0]])
            for k in df_metricas.index
            if cols_gap and not pd.isna(df_metricas.loc[k, cols_gap[0]])
        },
    }
    with open(caminhos['metrics'], 'w', encoding='utf-8') as f:
        json.dump(payload_metrics, f, indent=2, ensure_ascii=False)

    print(f"Artefatos do modelo final ({artefato['nome_modelo']}) exportados com sucesso em {dest}:")
    for c in caminhos.values():
        print(f" - {c.name}")
    return caminhos
