"""
03 | Avaliação In-Time vs. Out-of-Time (reprodução do 04_Avaliacao.ipynb)

.joblib + df_master -> escoragem (Treino OOF + Validação + OOT)
  -> métricas por janela, KS/decis, SHAP, target drift, PSI (score e features), GINI semanal
  -> desempenho por grupo de ataque (clusters do 03_Segmentacao / 05_segment_fraud)
  -> artifacts/monitoramento_oot.json + reports/metrics/avaliacao_*.csv

As figuras do 04 são geradas com as mesmas funções e exportadas para <reports>/figures
apenas quando `evaluation.save_figures: true`.

Uso:
    python scripts/03_evaluate_oot.py [--set evaluation.save_figures=true]
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import json  # noqa: E402

from src.config import Config, executar_script, salvar_json  # noqa: E402

ETAPA = 'evaluate'
ORDEM_GRUPOS_PADRAO = ['1. Cartão Clonado', '2. Botnet', '3. Roubo de Contas (ATO)', '4. Ataque Inteligente']


def _status_psi(psi: float, alerta: float, critico: float) -> str:
    return 'Estável' if psi < alerta else ('Alerta' if psi < critico else 'Crítico')


def avaliar_grupos(cfg: Config, df_master, meta, df_scores, faixa_corte, dir_met: Path):
    """Seção 8 do 04: recall/vazamento por grupo de ataque. Retorna (df_grupos, cobertura) ou (None, None)."""
    import pandas as pd
    from src.segmentation.fraud_clusters import obter_rotulos_clusters_fraude
    from src.evaluation.metrics import avaliar_modelo_por_grupo
    from src.visualization.eval_plots import plot_recall_grupos_janelas, plot_score_fraudes_por_grupo

    seg = cfg['segmentation']
    nome_cache = f"fraude_clusters_treino_k{int(seg['k'])}.parquet"
    cache = cfg.entrada('processed', nome_cache, obrigatorio=False)
    if not cache.exists() and not seg['enabled']:
        print(f"[evaluate] Cache de clusters ausente ({nome_cache}) e segmentation.enabled=false -> Seção 8 ignorada.")
        return None, None
    destino = cfg.saida('processed') / nome_cache
    if cache.exists() and cache != destino:
        # cache padrão (03_Segmentacao) lido por um experimento isolado: copia para não regravar o original
        import shutil
        shutil.copy2(cache, destino)
    cache = destino

    mapa = {int(k): v for k, v in seg['cluster_map'].items()}
    df_cl = obter_rotulos_clusters_fraude(
        df_master, meta, mapa, caminho_cache=cache, k=int(seg['k']), n_neighbors=int(seg['n_neighbors']),
        min_dist=float(seg['min_dist']), random_state=int(cfg['experiment']['random_state']),
        forcar_recalculo=False, train_ratio=cfg['data']['train_ratio'],
    )
    mapa_id = df_cl.set_index('TransactionID')['cluster']
    df_fraudes = df_scores[df_scores['isFraud'] == 1].copy()
    df_fraudes['cluster'] = df_fraudes['TransactionID'].map(mapa_id)
    df_fraudes['grupo'] = df_fraudes['cluster'].map(mapa)
    cobertura = float(df_fraudes['cluster'].notna().mean())
    if cobertura <= 0.99:
        print(f"[evaluate] ATENÇÃO: rótulos de cluster cobrem apenas {cobertura:.2%} das fraudes.")
    ordem = [g for g in ORDEM_GRUPOS_PADRAO if g in set(mapa.values())] or sorted(set(mapa.values()))
    df_grupos = avaliar_modelo_por_grupo(df_fraudes, col_grupo='grupo', ordem=ordem)
    plot_recall_grupos_janelas(df_grupos, ordem=ordem, filepath='04_13_recall_vazamento_grupos.png')
    plot_score_fraudes_por_grupo(df_fraudes, col_grupo='grupo', ordem=ordem, faixa_corte=faixa_corte,
                                 filepath='04_14_score_fraudes_grupos.png')
    df_grupos.to_csv(dir_met / 'avaliacao_modelo_por_grupo.csv', index=False)
    return df_grupos, cobertura


def run(cfg: Config) -> dict:
    import hashlib
    from datetime import datetime

    import joblib
    import numpy as np
    import pandas as pd
    from scipy.stats import ks_2samp, spearmanr

    from src.data.loader import preparar_dados_modelagem
    from src.models.calibrate import gerar_scores_oof_treino, threshold_calibrado_para_bruto
    from src.models.threshold import politica_financeira
    from src.evaluation.metrics import (
        escorar_particao, adicionar_bloco_temporal, tabela_metricas_janelas, gerar_tabela_decis_ks, tabela_temporal,
    )
    from src.evaluation.drift import teste_qui_quadrado_temporal, calcular_psi_scores, relatorio_psi_features
    from src.evaluation.explainability import calcular_shap_lgbm, ranking_shap
    from src.visualization.theme import configurar_exportacao_figuras
    from src.visualization.eval_plots import (
        plot_matrizes_confusao_janelas, plot_roc_pr_janelas, plot_kde_scores_janelas, plot_cdf_ks_janelas,
        plot_decis_janelas, plot_importancia_shap, plot_shap_beeswarm, plot_target_drift_temporal,
        plot_psi_score_faixas, plot_psi_vs_shap, plot_comparativo_psi_features, plot_gini_temporal,
    )

    ev, d, dp = cfg['evaluation'], cfg['data'], cfg['decision_policy']
    seed = int(cfg['experiment']['random_state'])
    dir_met, dir_art = cfg.saida('metrics'), cfg.saida('artifacts')
    configurar_exportacao_figuras(ativo=bool(ev['save_figures']))

    # 1.1 Carga
    df_master = pd.read_parquet(cfg.entrada('processed', 'df_master.parquet'))
    with open(cfg.entrada('processed', 'features_metadata.json'), 'r', encoding='utf-8') as f:
        meta = json.load(f)
    caminho_artefato = cfg.entrada('artifacts', 'modelo_fraude_final.joblib')
    artefato = joblib.load(caminho_artefato)
    FEATURES, REGRA = artefato['features'], artefato['regra_decisao']
    hash_artefato = hashlib.md5(caminho_artefato.read_bytes()).hexdigest()[:8]
    pf = REGRA.get('politica_financeira', {})
    mdr, perda = pf.get('mdr', dp['mdr_rate']), pf.get('perda_fraude', dp['fraud_loss_rate'])
    print(f"[evaluate] Artefato {artefato['nome_modelo']} ({len(FEATURES)} features, hash {hash_artefato}) <- {caminho_artefato}")

    with politica_financeira(mdr, perda):
        # 1.2 Split (mesma esteira do 02)
        dp_ = preparar_dados_modelagem(df_master=df_master, meta=meta, train_ratio=d['train_ratio'], test_ratio=d['val_ratio'])
        X_train, y_train = dp_['X_train'], dp_['y_train']
        X_val, X_oot = dp_['X_test'], dp_['X_oot']
        ausentes = [f_ for f_ in FEATURES if f_ not in X_oot.columns]
        if ausentes:
            raise ValueError(f"Features do artefato ausentes no OOT: {ausentes}")
        X_in_time = pd.concat([X_train[FEATURES], X_val[FEATURES]])
        X_oot_m = X_oot[FEATURES]

        # 1.3 Escoragem
        prob_raw_treino = (gerar_scores_oof_treino(artefato, X_train, y_train, groups=dp_['grupos_mes'],
                                                   n_splits=int(cfg['calibration']['oof_splits']))
                           if ev['oof_train_scoring'] else None)
        rotulo_treino = 'Treino (OOF)'
        df_scores = pd.concat([
            escorar_particao(artefato, X_train, dp_['train_df'], rotulo_treino, 'In-Time', prob_raw=prob_raw_treino),
            escorar_particao(artefato, X_val, dp_['test_df'], 'Validação', 'In-Time'),
            escorar_particao(artefato, X_oot, dp_['oot_df'], 'OOT', 'OOT'),
        ])
        df_scores = adicionar_bloco_temporal(df_scores, dt_inicio=df_master['TransactionDT'].min(),
                                             dias_por_bloco=int(ev['block_days']), nome_coluna='semana')
        base_treino = df_scores[df_scores['Particao'] == rotulo_treino]
        base_val = df_scores[df_scores['Particao'] == 'Validação']
        base_it = df_scores[df_scores['Janela'] == 'In-Time']
        base_oot = df_scores[df_scores['Janela'] == 'OOT']
        faixa_corte = (threshold_calibrado_para_bruto(artefato['calibrador'], REGRA['t_min']),
                       threshold_calibrado_para_bruto(artefato['calibrador'], REGRA['t_max']))

        # 2.0 Métricas por janela
        bases = {rotulo_treino: base_treino, 'Validação': base_val, 'In-Time': base_it, 'OOT': base_oot}
        df_metricas = tabela_metricas_janelas(bases, ref='In-Time', alvo='OOT')
        df_metricas.to_csv(dir_met / 'avaliacao_metricas_in_time_vs_oot.csv')
        plot_matrizes_confusao_janelas({'In-Time': base_it, 'Out-of-Time (OOT)': base_oot},
                                       filepath='04_01_matriz_confusao_in_time_vs_oot.png')
        plot_roc_pr_janelas(base_it['isFraud'], base_it['prob_raw'], base_oot['isFraud'], base_oot['prob_raw'],
                            filepath='04_02_roc_pr_in_time_vs_oot.png')

        # 4.0 Separação (KDE, KS, decis)
        plot_kde_scores_janelas(base_it['isFraud'], base_it['prob_raw'], base_oot['isFraud'], base_oot['prob_raw'],
                                faixa_corte=faixa_corte, calibrador=artefato['calibrador'],
                                filepath='04_03_kde_scores_in_time_vs_oot.png')
        ks_it, ks_oot = plot_cdf_ks_janelas(base_it['isFraud'], base_it['prob_raw'], base_oot['isFraud'],
                                            base_oot['prob_raw'], filepath='04_04_cdf_ks_in_time_vs_oot.png')
        n_decis = int(ev['n_decis'])
        tab_it = gerar_tabela_decis_ks(base_it['isFraud'], base_it['prob_raw'], n_decis=n_decis)
        tab_oot = gerar_tabela_decis_ks(base_oot['isFraud'], base_oot['prob_raw'], n_decis=n_decis)
        plot_decis_janelas(tab_it, tab_oot, filepath='04_05_decis_ganho_in_time_vs_oot.png')
        tab_oot.to_csv(dir_met / 'avaliacao_decis_oot.csv', index=False)
        monotonico_oot = bool(np.all(np.diff(tab_oot['taxa_fraude_pct']) >= 0))
        top = tab_oot.loc[tab_oot['decil'] == tab_oot['decil'].max()].iloc[0]

        # 5.0 SHAP
        shap_oot, X_shap_oot = calcular_shap_lgbm(artefato['modelo'], X_oot_m, n_amostra=int(ev['shap_sample_size']),
                                                  random_state=seed)
        shap_it, _ = calcular_shap_lgbm(artefato['modelo'], X_in_time, n_amostra=int(ev['shap_sample_size']),
                                        random_state=seed)
        rank_oot, rank_it = ranking_shap(shap_oot, FEATURES), ranking_shap(shap_it, FEATURES)
        plot_importancia_shap(rank_oot, top_n=15, n_destaque=3, titulo='Impacto Global das Variáveis no Risco de Fraude — OOT',
                              filepath='04_06_shap_importancia_oot.png')
        plot_shap_beeswarm(shap_oot, X_shap_oot, top_n=15, titulo='Direção do Impacto das Variáveis — OOT',
                           filepath='04_07_shap_beeswarm_oot.png')
        rank_oot.to_csv(dir_met / 'avaliacao_shap_oot.csv', index=False)
        _m = rank_oot[['Feature', 'SHAP_medio_abs']].merge(rank_it[['Feature', 'SHAP_medio_abs']], on='Feature',
                                                            suffixes=('_OOT', '_InTime'))
        spearman_shap = float(spearmanr(_m['SHAP_medio_abs_InTime'], _m['SHAP_medio_abs_OOT']).correlation)

        # 6.1 Target drift
        df_temporal = tabela_temporal(df_scores, col_bloco='semana', min_fraudes=int(ev['min_fraudes_bloco']),
                                      min_transacoes=int(ev['min_transacoes_bloco']))
        medias_taxa = plot_target_drift_temporal(df_temporal, col_bloco='semana', rotulo_bloco='Semana',
                                                 filepath='04_08_target_drift_semanal.png')
        teste_qui_quadrado_temporal(df_temporal, rotulo='semanas (heterogeneidade temporal)')
        tab_janelas = df_scores.groupby('Janela').agg(Fraudes=('isFraud', 'sum'), Transacoes=('isFraud', 'size'))
        chi2_janelas = teste_qui_quadrado_temporal(tab_janelas, rotulo='In-Time x OOT')
        df_temporal.to_csv(dir_met / 'avaliacao_metricas_semanais.csv', index=False)

        # 6.2 PSI do score
        bins = int(ev['psi_bins'])
        lim = ev['psi_thresholds']
        df_psi_score, psi_score = calcular_psi_scores(base_it['prob_raw'], base_oot['prob_raw'], bins=bins)
        ks_shift = ks_2samp(base_it['prob_raw'], base_oot['prob_raw'])
        plot_psi_score_faixas(df_psi_score, psi_score, filepath='04_09_psi_score_faixas.png')

        # 6.3 PSI das features x SHAP
        df_psi_features = relatorio_psi_features(X_in_time, X_oot_m, features=FEATURES, bins=bins)
        df_psi_features.to_csv(dir_met / 'avaliacao_psi_features.csv', index=False)
        df_drift_risco = plot_psi_vs_shap(df_psi_features, rank_oot, n_rotulos=8, filepath='04_10_matriz_psi_vs_shap.png')
        plot_comparativo_psi_features(X_in_time, X_oot_m, df_drift_risco['Feature'].head(3).tolist(), bins=bins,
                                      filepath='04_11_feature_drift_top3.png')

        # 7.0 GINI semanal
        d_it = df_temporal[df_temporal['Janela'] == 'In-Time'].dropna(subset=['GINI'])
        gini_medio_it = float(np.average(d_it['GINI'], weights=d_it['Transacoes']))
        piso_gini = gini_medio_it * (1 - float(ev['gini_drop_tolerance']))
        medias_gini = plot_gini_temporal(df_temporal, col_bloco='semana', rotulo_bloco='Semana', piso_gini=piso_gini,
                                         filepath='04_12_gini_temporal_semanal.png')
        d_oot = df_temporal[df_temporal['Janela'] == 'OOT'].dropna(subset=['GINI'])
        semanas_abaixo = d_oot.loc[d_oot['GINI'] < piso_gini, 'semana'].tolist()
        tendencia_oot = float(np.polyfit(d_oot['semana'], d_oot['GINI'], 1)[0]) if len(d_oot) >= 3 else None

        # 8.0 Grupos de ataque
        df_grupos, cobertura = avaliar_grupos(cfg, df_master, meta, df_scores, faixa_corte, dir_met)

    # 9.0 Monitoramento (MLOps)
    grupos = None
    if df_grupos is not None:
        _oot = df_grupos[df_grupos['Janela'] == 'OOT']
        pior = _oot.loc[_oot['Prejuizo_Vazado_USD'].idxmax()]
        grupos = {'por_grupo': df_grupos.astype({'Grupo': str}).to_dict(orient='records'),
                  'cobertura_rotulos': cobertura, 'pior_grupo_oot': str(pior['Grupo']),
                  'pior_grupo_oot_vazamento_usd': float(pior['Prejuizo_Vazado_USD'])}

    monitoramento = {
        'timestamp_avaliacao': datetime.now().isoformat(timespec='seconds'),
        'experimento': cfg['experiment']['name'],
        'modelo': artefato['nome_modelo'],
        'hash_artefato': hash_artefato,
        'total_features': len(FEATURES),
        'regra_decisao': REGRA,
        'treino_escorado_oof': bool(ev['oof_train_scoring']),
        'metricas_por_base': {b: df_metricas[b].to_dict() for b in bases},
        'delta_oot_vs_in_time': df_metricas['Δ (OOT − In-Time)'].dropna().to_dict(),
        'separacao': {'ks_in_time': ks_it, 'ks_oot': ks_oot, 'monotonico_oot': monotonico_oot,
                      'lift_decil10_oot': float(top['lift']), 'captura_decil10_oot_pct': float(top['captura_fraudes_acum_pct'])},
        'drift': {
            'taxa_fraude_media': medias_taxa,
            'qui2_in_time_vs_oot': chi2_janelas,
            'psi_score': psi_score,
            'status_psi_score': _status_psi(psi_score, lim['alert'], lim['critical']),
            'ks_shift_score': float(ks_shift.statistic),
            'psi_features': df_psi_features.set_index('Feature')['PSI'].to_dict(),
            'top_risco_drift': df_drift_risco.head(10)[['Feature', 'PSI', 'Risco_Drift']].to_dict(orient='records'),
        },
        'shap': {'spearman_in_time_vs_oot': spearman_shap,
                 'top15_oot': rank_oot.head(15)[['Feature', 'SHAP_medio_abs']].to_dict(orient='records')},
        'estabilidade_temporal': {
            'gini_medio': medias_gini, 'piso_gini': piso_gini, 'semanas_oot_abaixo_piso': semanas_abaixo,
            'tendencia_gini_oot_semana': tendencia_oot,
            'serie_semanal': df_temporal[['semana', 'Particao', 'Taxa_Fraude_%', 'GINI', 'Recall_%']].to_dict(orient='records'),
        },
        'grupos': grupos,
    }
    caminho = salvar_json(monitoramento, dir_art / 'monitoramento_oot.json')
    print(f"[evaluate] Monitoramento exportado: {caminho}")
    return {'monitoramento': str(caminho)}


if __name__ == '__main__':
    executar_script(ETAPA, __doc__, run)
