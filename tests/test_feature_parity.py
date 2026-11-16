"""
Testes rápidos de sanidade (executar com `python -m pytest tests -q`).

- Paridade de features: `aplicar_fe_ieee_cis_base` gera o MESMO schema/escala no treino e no test;
- Fachada `utils.py`: re-exporta todos os nomes de `src/` (compatibilidade dos notebooks);
- Exportação de figuras: tudo vai para um diretório plano (sem subpastas);
- Política financeira: o padrão (2% / 100%) não é alterado fora do contexto do configs.yaml;
- Split temporal: 60/20/20 cronológico com frequências ajustadas só no Treino;
- Config: overrides, validação e isolamento de experimentos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# --------------------------------------------------------------------------- FE
def test_paridade_schema_treino_vs_test(bases_treino, bases_teste):
    from src.features.engineering import aplicar_fe_ieee_cis_base

    tr, idf = bases_treino
    te, idf_te = bases_teste
    df_tr, _ = aplicar_fe_ieee_cis_base(tr.drop(columns=['isFraud']), idf)
    df_te, ctx = aplicar_fe_ieee_cis_base(te, idf_te, n_ref=len(tr))

    assert list(df_tr.columns) == list(df_te.columns), set(df_tr.columns) ^ set(df_te.columns)
    assert not any(c.startswith('id-') for c in df_te.columns), 'id-XX deveria virar id_XX'
    for c in df_tr.columns:
        num_tr = pd.api.types.is_numeric_dtype(df_tr[c])
        num_te = pd.api.types.is_numeric_dtype(df_te[c])
        assert num_tr == num_te, f'{c}: numérica no treino={num_tr} x test={num_te}'
    assert len(ctx['TransactionID']) == len(te)
    np.testing.assert_allclose(ctx['amt_usd'], te['TransactionAmt'].values)


def test_fe_treino_descarta_tempo_dia_corrido(bases_treino):
    from src.features.engineering import aplicar_fe_treino

    tr, idf = bases_treino
    df = aplicar_fe_treino(tr, idf)
    assert 'tempo_dia_corrido' not in df.columns
    for c in ['tempo_hora', 'valor_centavos', 'cont_null_tr', 'match_cont_T', 'cartao_bandeira_tipo',
              'vel_C1', 'delta_D1', 'cont_null_id', 'disp_marca', 'freq_card1', 'valor_razao_media_card1']:
        assert c in df.columns, c
    assert {'C1', 'D1', 'V305', 'id_07', 'id_31', 'DeviceInfo'}.isdisjoint(df.columns)
    assert len(df) == len(tr), 'merge left não pode duplicar/perder transações'


def test_freq_reescalada_no_test(bases_treino, bases_teste):
    from src.features.engineering import aplicar_fe_ieee_cis_base

    te, idf_te = bases_teste
    n_ref = len(bases_treino[0])
    df_te, _ = aplicar_fe_ieee_cis_base(te, idf_te, n_ref=n_ref)
    bruto = te['card1'].map(te['card1'].value_counts(dropna=False))
    np.testing.assert_allclose(df_te['freq_card1'].values, (bruto * n_ref / len(te)).values, rtol=1e-5)


def test_freq_categorica_ausente_none_vs_nan(bases_treino, bases_teste, tmp_path):
    """Regressão: e-mail ausente como `None` (parquet do df_master) x `NaN` (CSV do Test) deve ter a mesma frequência."""
    from src.features.engineering import aplicar_fe_treino, preparar_dados_teste_inferencia

    tr, idf = bases_treino
    te, idf_te = bases_teste
    te.to_csv(tmp_path / 't.csv', index=False)
    te = pd.read_csv(tmp_path / 't.csv')                     # mesmo caminho do 05: ausente vira NaN
    master = aplicar_fe_treino(tr, idf)
    master.to_parquet(tmp_path / 'm.parquet', index=False)
    master = pd.read_parquet(tmp_path / 'm.parquet')        # mesmo caminho dos notebooks: ausente vira None
    assert master['P_emaildomain'].isna().any()

    feats = ['P_emaildomain_freq', 'card1_freq']
    prep = preparar_dados_teste_inferencia(te, idf_te, master, feats, meta={}, train_ratio=0.6, test_ratio=0.2)
    treino = master.sort_values('TransactionDT').iloc[:int(len(master) * 0.6)]
    n_ausentes_treino = int(treino['P_emaildomain'].isna().sum())
    ausente_test = te['P_emaildomain'].isna().values
    assert ausente_test.any()
    np.testing.assert_allclose(prep['X_test'].loc[ausente_test, 'P_emaildomain_freq'], n_ausentes_treino)


# --------------------------------------------------------------------- fachada
def test_fachada_utils_reexporta_src():
    import importlib
    import utils

    faltando = [n for n in utils.__all__ if not hasattr(utils, n)]
    assert not faltando, faltando
    from src.models.artifacts import aplicar_regra_dinamica
    from src.visualization.theme import PALETA_AZUL
    assert utils.aplicar_regra_dinamica is aplicar_regra_dinamica
    assert utils.PALETA_AZUL is PALETA_AZUL
    assert utils.analise_correlação_pares is utils.analise_correlacao_pares
    importlib.reload(utils)  # padrão dos notebooks
    assert hasattr(utils, '_salvar_fig') and hasattr(utils, 'preparar_dados_modelagem')


# ------------------------------------------------------------------- figuras
def test_exportacao_figuras_diretorio_plano(tmp_path):
    import matplotlib.pyplot as plt
    from src.visualization import theme

    anterior = theme.configurar_exportacao_figuras()
    try:
        theme.configurar_exportacao_figuras(ativo=True, diretorio=tmp_path)
        fig, ax = plt.subplots()
        ax.set_title('Curva ROC — In-Time')
        theme._salvar_fig(fig, filepath=tmp_path / 'sub' / 'pasta' / '02_roc.png')
        assert (tmp_path / '02_roc.png').exists()
        assert not (tmp_path / 'sub').exists(), 'não deve criar subpastas'

        fig2, ax2 = plt.subplots()
        ax2.set_title('Distribuição do Alvo')
        theme._mostrar_figura()
        gerados = sorted(p.name for p in tmp_path.iterdir())
        assert any(n.endswith('__distribuicao_do_alvo.png') for n in gerados), gerados

        fig3, _ = plt.subplots()
        theme._salvar_figura_opcional(fig3, nome_arquivo=None, save_dir=tmp_path / 'ignorado')
        assert not (tmp_path / 'ignorado').exists()
        assert all(p.is_file() for p in tmp_path.iterdir())
    finally:
        theme.configurar_exportacao_figuras(ativo=anterior['ativo'], diretorio=anterior['diretorio'])


def test_exportacao_desativada_nao_grava(tmp_path):
    import matplotlib.pyplot as plt
    from src.visualization import theme

    anterior = theme.configurar_exportacao_figuras()
    try:
        theme.configurar_exportacao_figuras(ativo=False, diretorio=tmp_path)
        fig, _ = plt.subplots()
        theme._salvar_fig(fig, 'x.png')
        assert list(tmp_path.iterdir()) == []
    finally:
        theme.configurar_exportacao_figuras(ativo=anterior['ativo'], diretorio=anterior['diretorio'])


# ------------------------------------------------------- política financeira
def test_politica_financeira_padrao_e_contexto():
    from src.models.threshold import (
        calcular_matriz_lucro, curva_lucro_threshold, politica_financeira, taxas_financeiras,
    )

    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.1).astype(int)
    p = rng.random(500)
    amt = rng.lognormal(4, 1, 500)
    bloq = p >= 0.3
    esperado = (0.02 * amt[(~bloq) & (y == 0)]).sum() - (0.02 * amt[bloq & (y == 0)]).sum() - amt[(~bloq) & (y == 1)].sum()

    assert taxas_financeiras() == (0.02, 1.00)
    assert np.isclose(calcular_matriz_lucro(y, p, 0.3, amt)['Lucro'], esperado)
    assert np.isclose(curva_lucro_threshold(y, p, amt, [0.3])['Lucro'].iloc[0], esperado)
    with politica_financeira(0.03, 0.5):
        assert taxas_financeiras() == (0.03, 0.5)
        assert not np.isclose(calcular_matriz_lucro(y, p, 0.3, amt)['Lucro'], esperado)
    assert taxas_financeiras() == (0.02, 1.00), 'contexto deve restaurar o padrão dos notebooks'


def test_regra_fixa_equivale_threshold_constante():
    from src.models.threshold import regra_threshold_fixo, threshold_dinamico

    r = regra_threshold_fixo(0.04)
    np.testing.assert_allclose(threshold_dinamico([1, 100, 5000], r['t_min'], r['t_max'], r['k']), 0.04)


def test_calibrador_platt_interface():
    from src.models.calibrate import CalibradorPlatt, threshold_calibrado_para_bruto

    rng = np.random.default_rng(0)
    raw = rng.random(2000)
    y = (rng.random(2000) < raw * 0.2).astype(int)
    cal = CalibradorPlatt().fit(raw, y)
    p = cal.transform(raw)
    assert p.min() >= 0 and p.max() <= 1
    bruto = threshold_calibrado_para_bruto(cal, 0.05)
    assert np.isclose(cal.transform([bruto])[0], 0.05, atol=1e-6)


# --------------------------------------------------------------------- split
def test_split_temporal_sem_leakage(bases_treino):
    from src.data.loader import preparar_dados_modelagem
    from src.features.engineering import aplicar_fe_treino

    tr, idf = bases_treino
    df = aplicar_fe_treino(tr, idf)
    d = preparar_dados_modelagem(df, meta={}, train_ratio=0.60, test_ratio=0.20)
    n = len(df)
    assert len(d['X_train']) == int(n * 0.6) and len(d['X_oot']) == n - int(n * 0.8)
    assert d['train_df']['TransactionDT'].max() <= d['test_df']['TransactionDT'].min()
    assert d['test_df']['TransactionDT'].max() <= d['oot_df']['TransactionDT'].min()
    freq_treino = d['train_df']['card1'].map(d['train_df']['card1'].value_counts(dropna=False))
    np.testing.assert_allclose(d['X_train']['card1_freq'].values, freq_treino.values)
    assert {'card1', 'TransactionID', 'isFraud', 'TransactionDT'}.isdisjoint(d['features_cols'])


# -------------------------------------------------------------------- config
def test_config_overrides_validacao_e_isolamento():
    from src.config import PROJECT_ROOT, carregar_config

    cfg = carregar_config(overrides=['model.tuning.n_trials=3', 'experiment.name=teste_ci',
                                     'decision_policy.bounds.t_max=[0.03, 0.15]'])
    assert cfg['model']['tuning']['n_trials'] == 3
    assert cfg['decision_policy']['bounds']['t_max'] == [0.03, 0.15]
    assert cfg.isolado and cfg.raiz_saida == PROJECT_ROOT / 'experiments' / 'teste_ci'
    assert cfg.saida('figures', criar=False) == PROJECT_ROOT / 'experiments' / 'teste_ci' / 'reports' / 'figures'
    assert cfg.saida('raw', criar=False) == PROJECT_ROOT / 'data' / 'raw'

    padrao = carregar_config(overrides=['experiment.overwrite_default_artifacts=true'])
    assert padrao.saida('artifacts', criar=False) == PROJECT_ROOT / 'artifacts'

    with pytest.raises(ValueError):
        carregar_config(overrides=['model.estimator=randomforest'])
    with pytest.raises(ValueError):
        carregar_config(overrides=['data.train_ratio=0.9'])


def test_notebooks_nao_leem_configs():
    """As funções de src/ (usadas pelos notebooks) não dependem do configs.yaml."""
    from pathlib import Path

    raiz = Path(__file__).resolve().parents[1]
    arquivos = [raiz / 'utils.py'] + [p for p in (raiz / 'src').rglob('*.py') if p.name != 'config.py']
    for p in arquivos:
        texto = p.read_text(encoding='utf-8')
        assert 'carregar_config' not in texto, f'{p} não deve ler o configs.yaml'
        assert 'import yaml' not in texto, f'{p} não deve ler YAML'
