"""
01 | Preparação dos dados (reprodução do 01_EDA.ipynb sem gráficos)

Raw (train_transaction + train_identity)
  -> Feature Engineering (fonte única `aplicar_fe_ieee_cis_base`)
  -> Caudas longas (log1p, skew > limiar)
  -> Information Value
  -> Funil de seleção (brutas, freq_* globais, IV mínimo, correlação, singularidade V240/V241)
  -> df_master.parquet + features_metadata.json (+ relatórios em reports/metrics)

Uso:
    python scripts/01_prepare_data.py [--config configs.yaml] [--set features.iv_min=0.03]
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import gc  # noqa: E402

from src.config import Config, executar_script, salvar_json  # noqa: E402

ETAPA = 'prepare'


def run(cfg: Config) -> dict:
    from src.data.loader import garantir_bases_brutas, carregar_bases_brutas
    from src.data.preprocess import analisar_e_transformar_caudas_longas
    from src.features.engineering import aplicar_fe_treino
    from src.features.selection import calcular_iv_dataset, funil_selecao_df_master, analisar_vif

    p, f, d = cfg['paths'], cfg['features'], cfg['data']
    seed = int(cfg['experiment']['random_state'])
    dir_proc, dir_met = cfg.saida('processed'), cfg.saida('metrics')

    if not f['apply_fe']:
        caminho = cfg.entrada('processed', 'df_master.parquet')
        print(f"[prepare] features.apply_fe=false -> mantendo df_master existente: {caminho}")
        return {'df_master': str(caminho), 'pulado': True}

    # 1. Raw
    raw_dir = garantir_bases_brutas(cfg.saida('raw'), auto_download=p['auto_download_kaggle'],
                                    competicao=p['kaggle_competition'])
    df_tr, df_id = carregar_bases_brutas(raw_dir, 'train')

    # 2. Feature Engineering (01_EDA Seções 3.2–6.2)
    df_train = aplicar_fe_treino(df_tr, df_id)
    del df_tr, df_id
    gc.collect()

    # 3. Caudas longas (01_EDA Seção 6.3)
    df_train, df_skew = analisar_e_transformar_caudas_longas(
        df_train, limiar_skew=f['log1p_skew_threshold'], min_unicos=f['log1p_min_unique'],
        substituir_original=True, plotar_top=0,
    )
    cols_log1p = df_skew.loc[df_skew['Status'].str.startswith('Transformada'), 'Feature'].tolist()
    df_skew.to_csv(dir_met / 'prepare_caudas_longas_log1p.csv', index=False)

    # 4. Information Value (01_EDA Seção 7.1)
    df_iv = calcular_iv_dataset(df_train, target_col='isFraud', bins=f['iv_bins'])
    df_iv.to_csv(dir_met / 'prepare_information_value.csv', index=False)

    # 5. Funil de seleção (01_EDA Seções 7.2.1–7.4)
    df_master, metadata, resumo = funil_selecao_df_master(
        df_train, df_iv, iv_min=f['iv_min'], corr_max=f['corr_max'], tolerancia_nulos=f['corr_null_tolerance'],
        linear_top_n=f['linear_baseline_top_n'], linear_max_nulos=f['linear_baseline_max_null'], downcast=d['downcast'],
    )
    del df_train
    gc.collect()

    # Informativo (não consumido pelos notebooks): colunas transformadas em log1p no treino
    metadata['colunas_log1p_treino'] = [c for c in cols_log1p if c in df_master.columns]

    # 6. VIF (diagnóstico — 01_EDA Seção 7.3)
    if f.get('vif_max') is not None:
        df_vif = analisar_vif(df_master, limite=float(f['vif_max']), imputar_nulos=True,
                              sample_size=f['vif_sample_size'], random_state=seed, target_col='isFraud')
        df_vif.to_csv(dir_met / 'prepare_vif.csv', index=False)

    # 7. Exportação
    caminho_master = dir_proc / 'df_master.parquet'
    df_master.to_parquet(caminho_master, index=False)
    salvar_json(metadata, dir_proc / 'features_metadata.json')
    salvar_json(resumo, dir_met / 'prepare_resumo_funil.json')
    print(f"[prepare] df_master: {df_master.shape} -> {caminho_master}")
    return {'df_master': str(caminho_master), 'shape': list(df_master.shape)}


if __name__ == '__main__':
    executar_script(ETAPA, __doc__, run)
