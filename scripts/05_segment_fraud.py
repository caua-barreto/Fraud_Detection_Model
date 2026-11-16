"""
05 | (Opcional) Segmentação das fraudes (reprodução das Seções 1–6 do 03_Segmentacao.ipynb)

df_master -> fraudes do Treino (60%) -> preparo (encoding + RobustScaler) -> UMAP 10D -> K-Means (K=12)
  -> mapeamento cluster -> arquétipo (segmentation.cluster_map)
  -> projeção das fraudes de Validação/OOT
  -> data/processed/fraude_clusters_treino_k<K>.parquet (cache lido pelo 03_evaluate_oot / 04_Avaliacao)
  -> reports/metrics/segmentacao_*.csv

Requer `umap-learn`. Uso:
    python scripts/05_segment_fraud.py [--set segmentation.k=10]
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import json  # noqa: E402

from src.config import Config, executar_script, salvar_json  # noqa: E402

ETAPA = 'segment'
# Sanidade documentada no 03/04: produto dominante esperado por arquétipo
PRODUTO_ESPERADO = {
    '1. Cartão Clonado': ('W',),
    '2. Botnet': ('C',),
    '3. Roubo de Contas (ATO)': ('R', 'H', 'C'),
    '4. Ataque Inteligente': ('H',),
}


def run(cfg: Config) -> dict:
    import pandas as pd

    from src.segmentation.fraud_clusters import (
        obter_rotulos_clusters_fraude, recortar_treino_temporal, gerar_perfil_fraude_clusters, gerar_tabela_arquetipos,
    )

    seg = cfg['segmentation']
    if not seg['enabled']:
        print('[segment] segmentation.enabled=false -> etapa ignorada.')
        return {'pulado': True}

    k = int(seg['k'])
    mapa = {int(c): g for c, g in seg['cluster_map'].items()}
    dir_proc, dir_met = cfg.saida('processed'), cfg.saida('metrics')
    df_master = pd.read_parquet(cfg.entrada('processed', 'df_master.parquet'))
    with open(cfg.entrada('processed', 'features_metadata.json'), 'r', encoding='utf-8') as f:
        meta = json.load(f)

    cache = dir_proc / f'fraude_clusters_treino_k{k}.parquet'
    df_cl = obter_rotulos_clusters_fraude(
        df_master, meta, mapa, caminho_cache=cache, k=k, n_neighbors=int(seg['n_neighbors']),
        min_dist=float(seg['min_dist']), random_state=int(cfg['experiment']['random_state']),
        forcar_recalculo=bool(seg['force_recompute']), train_ratio=cfg['data']['train_ratio'],
    )
    sem_mapa = sorted(set(df_cl['cluster'].unique()) - set(mapa))
    if sem_mapa:
        print(f"[segment] ATENÇÃO: clusters sem arquétipo em segmentation.cluster_map: {sem_mapa}")

    # Perfilamento das fraudes do Treino (Seções 5.1 e 6.1)
    df_treino = recortar_treino_temporal(df_master, train_ratio=cfg['data']['train_ratio'])
    rot_treino = df_cl[df_cl['base_cluster'] == 'Treino (ajuste)'][['TransactionID', 'cluster', 'arquetipo']]
    df_fraude = df_treino[df_treino['isFraud'] == 1].merge(rot_treino, on='TransactionID', how='inner')
    df_fraude = df_fraude.sort_values('TransactionDT').reset_index(drop=True)

    perfil = gerar_perfil_fraude_clusters(df_fraude, col_cluster='cluster', ignorar_ruido=False) \
        .sort_values('Prejuizo_Total_USD', ascending=False)
    perfil.to_csv(dir_met / f'segmentacao_perfil_clusters_k{k}.csv', index=False)

    ordem = [g for g in PRODUTO_ESPERADO if g in set(mapa.values())] or sorted(set(mapa.values()))
    tab_arq = gerar_tabela_arquetipos(df_fraude.dropna(subset=['arquetipo']), col_grupo='arquetipo', ordem=ordem,
                                      col_cluster_origem='cluster')
    tab_arq.to_csv(dir_met / 'segmentacao_tabela_arquetipos.csv', index=False)

    dominante = df_fraude.groupby('arquetipo')['ProductCD'].agg(lambda s: s.astype(str).value_counts().idxmax())
    divergentes = {g: dominante.get(g) for g, esp in PRODUTO_ESPERADO.items()
                   if g in dominante.index and dominante.get(g) not in esp}
    if divergentes:
        print(f"[segment] ATENÇÃO: produto dominante diverge da assinatura documentada {divergentes} "
              "-> revise segmentation.cluster_map (IDs de cluster podem mudar com outra versão do UMAP).")
    else:
        print('[segment] [OK] Produto dominante de cada arquétipo coerente com a assinatura documentada.')

    tamanhos = df_fraude['cluster'].value_counts().sort_index().to_dict()
    salvar_json({'k': k, 'tamanhos_cluster_treino': tamanhos, 'produto_dominante': dominante.to_dict(),
                 'divergencias_produto': divergentes, 'clusters_sem_mapa': sem_mapa,
                 'n_projetadas_val_oot': int((df_cl['base_cluster'] != 'Treino (ajuste)').sum())},
                dir_met / 'segmentacao_resumo.json')
    return {'cache_clusters': str(cache)}


if __name__ == '__main__':
    executar_script(ETAPA, __doc__, run)
