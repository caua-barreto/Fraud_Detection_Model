"""
04 | Escoragem cega do Test Kaggle (reprodução do 05_Submissao.ipynb)

.joblib + df_master + test_raw
  -> FE com paridade (aplicar_fe_ieee_cis_base + alinhamento log1p + encoders/freq do Treino)
  -> PSI das features (In-Time x Test) e PSI do score (Validação x Test)
  -> prob_raw -> prob_cal -> threshold dinâmico -> decisão
  -> data/submissions/submission_kaggle.csv (+ predicoes_test_binarias.csv) + reports/metrics/test_resumo_escoragem.json

Uso:
    python scripts/04_predict_test.py [--set inference.kaggle_float_precision=6]
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import gc  # noqa: E402
import json  # noqa: E402

from src.config import Config, executar_script, salvar_json, PROJECT_ROOT  # noqa: E402

ETAPA = 'predict'


def _rel(caminho: Path) -> str:
    try:
        return str(caminho.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(caminho)


def run(cfg: Config) -> dict:
    import joblib
    import numpy as np
    import pandas as pd

    from src.data.loader import garantir_bases_brutas, carregar_bases_brutas, ARQUIVOS_TESTE
    from src.features.engineering import preparar_dados_teste_inferencia
    from src.models.artifacts import aplicar_regra_dinamica
    from src.evaluation.metrics import escorar_particao
    from src.evaluation.drift import relatorio_psi_vs_importancia_modelo, calcular_psi_scores

    p, d, inf, ev = cfg['paths'], cfg['data'], cfg['inference'], cfg['evaluation']
    dir_sub, dir_met = cfg.saida('submissions'), cfg.saida('metrics')

    artefato = joblib.load(cfg.entrada('artifacts', 'modelo_fraude_final.joblib'))
    FEATURES = list(artefato['features'])
    caminho_meta = cfg.entrada('processed', 'features_metadata.json', obrigatorio=False)
    meta = json.loads(caminho_meta.read_text(encoding='utf-8')) if caminho_meta.exists() else None
    df_master = pd.read_parquet(cfg.entrada('processed', 'df_master.parquet'))

    raw_dir = garantir_bases_brutas(cfg.saida('raw'), ARQUIVOS_TESTE, auto_download=p['auto_download_kaggle'],
                                    competicao=p['kaggle_competition'], forcar_download=inf['force_download'])
    df_test_tr, df_test_id = carregar_bases_brutas(raw_dir, 'test')

    # 2.1 FE com paridade (mesma função do notebook 05)
    prep = preparar_dados_teste_inferencia(df_test_tr=df_test_tr, df_test_id=df_test_id, df_master=df_master,
                                           features_modelo=FEATURES, meta=meta,
                                           train_ratio=d['train_ratio'], test_ratio=d['val_ratio'])
    del df_test_tr, df_test_id, df_master
    gc.collect()
    X_test, amt_test = prep['X_test'], prep['amt_usd']
    assert list(X_test.columns) == FEATURES, 'Ordem/conjunto de colunas de X_test difere do artefato.'
    assert np.isfinite(amt_test).all(), 'amt_usd contém NaN ou infinito.'
    if not prep['df_oov'].empty:
        prep['df_oov'].to_csv(dir_met / 'test_auditoria_oov.csv', index=False)

    # 3.1 PSI das features x importância
    df_psi = relatorio_psi_vs_importancia_modelo(prep['X_in_time'], X_test, artefato, bins=int(ev['psi_bins']))
    df_psi.to_csv(dir_met / 'test_psi_features.csv', index=False)
    cont_status = df_psi['Status'].value_counts().reindex(['Estável', 'Alerta', 'Crítico'], fill_value=0).to_dict()

    # 4.1 Escoragem
    res = aplicar_regra_dinamica(artefato, X_test, amt_test)
    df_pred = pd.DataFrame({
        'TransactionID': prep['TransactionID'], 'TransactionDT': prep['TransactionDT'], 'ProductCD': prep['ProductCD'],
        'amt_usd': amt_test,
        'prob_raw': np.asarray(res['prob_raw'], dtype='float64'),
        'prob_cal': np.asarray(res['prob_cal'], dtype='float64'),
        'threshold_dinamico': np.asarray(res['threshold'], dtype='float64'),
        'pred_fraude': np.asarray(res['bloquear'], dtype='int8'),
    })
    df_val_sc = escorar_particao(artefato, prep['X_val'], prep['val_df'], 'Validação', 'In-Time')
    df_oot_sc = escorar_particao(artefato, prep['X_oot'], prep['oot_df'], 'OOT', 'OOT')

    # 4.2 PSI do score
    df_psi_score, psi_score = calcular_psi_scores(df_val_sc['prob_cal'], df_pred['prob_cal'], bins=int(ev['psi_bins']))
    share_d9_d10_test = float(df_psi_score.tail(2)['OOT_%'].sum() * 100)

    # 6.1 Exportação + validações
    arq_kaggle = dir_sub / 'submission_kaggle.csv'
    casas = int(inf['kaggle_float_precision'])
    df_kaggle = pd.DataFrame({'TransactionID': df_pred['TransactionID'].astype('int64'),
                              'isFraud': df_pred['prob_cal'].astype('float64')})
    df_kaggle.to_csv(arq_kaggle, index=False, float_format=f'%.{casas}f')
    chk = pd.read_csv(arq_kaggle)
    assert list(chk.columns) == ['TransactionID', 'isFraud'], 'Cabeçalho inválido na submissão Kaggle.'
    assert len(chk) == len(df_pred) and chk['TransactionID'].is_unique, 'Linhas/IDs inconsistentes.'
    assert not chk.isna().any().any() and chk['isFraud'].between(0.0, 1.0).all(), 'Scores inválidos.'
    arq_sample = raw_dir / 'sample_submission.csv'
    if arq_sample.exists():
        ids = pd.read_csv(arq_sample, usecols=['TransactionID'])['TransactionID'].values
        assert (chk['TransactionID'].values == ids).all(), 'TransactionID fora de ordem vs. sample_submission.csv.'

    arq_bin = None
    if inf['save_binary_audit']:
        arq_bin = dir_sub / 'predicoes_test_binarias.csv'
        df_pred.assign(TransactionID=df_pred['TransactionID'].astype('int64')).to_csv(arq_bin, index=False, float_format='%.6f')

    vol_total = float(df_pred['amt_usd'].sum())
    vol_bloq = float(df_pred.loc[df_pred['pred_fraude'] == 1, 'amt_usd'].sum())
    resumo = {
        'experimento': cfg['experiment']['name'],
        'modelo': str(artefato.get('nome_modelo')),
        'n_features_modelo': len(FEATURES),
        'n_transacoes_test': int(len(df_pred)),
        'prob_cal_media_val_pct': round(float(df_val_sc['prob_cal'].mean() * 100), 4),
        'prob_cal_media_oot_pct': round(float(df_oot_sc['prob_cal'].mean() * 100), 4),
        'prob_cal_media_test_pct': round(float(df_pred['prob_cal'].mean() * 100), 4),
        'taxa_bloqueio_val_pct': round(float(df_val_sc['bloquear'].mean() * 100), 4),
        'taxa_bloqueio_oot_pct': round(float(df_oot_sc['bloquear'].mean() * 100), 4),
        'taxa_bloqueio_test_pct': round(float(df_pred['pred_fraude'].mean() * 100), 4),
        'n_bloqueios_test': int(df_pred['pred_fraude'].sum()),
        'gmv_total_test_usd': round(vol_total, 2),
        'gmv_retido_test_usd': round(vol_bloq, 2),
        'pct_gmv_retido_test': round(vol_bloq / max(vol_total, 1e-9) * 100, 4),
        'psi_score_val_vs_test': round(float(psi_score), 5),
        'share_d9_d10_test_pct': round(share_d9_d10_test, 4),
        'features_psi_estavel': int(cont_status.get('Estável', 0)),
        'features_psi_alerta': int(cont_status.get('Alerta', 0)),
        'features_psi_critico': int(cont_status.get('Crítico', 0)),
        'arquivo_kaggle': _rel(arq_kaggle),
        'arquivo_binario': _rel(arq_bin) if arq_bin else None,
    }
    salvar_json(resumo, dir_met / 'test_resumo_escoragem.json')
    print(f"[predict] Submissão: {arq_kaggle} ({len(df_kaggle):,} linhas) | bloqueio {resumo['taxa_bloqueio_test_pct']:.2f}%")
    return {'submission': str(arq_kaggle)}


if __name__ == '__main__':
    executar_script(ETAPA, __doc__, run)
