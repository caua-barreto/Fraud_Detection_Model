"""Engenharia de atributos IEEE-CIS: fonte única para treino (01) e inferência (05).

Código migrado de `utils.py` (fachada de compatibilidade dos notebooks).
"""
from __future__ import annotations

import gc
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from src.data.loader import (  # noqa: F401
    preparar_dados_modelagem,
    safe_downcast,
)
from src.data.preprocess import (  # noqa: F401
    _detectar_e_aplicar_log1p_referencia,
    tratar_nulos_id,
)


def aplicar_feature_engineering_tr(df_tr: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica o pipeline de Feature Engineering sobre o DataFrame de transações (`df_train_tr`):
    1. Bloco M: contagem de 'T' e 'F', assinatura M1_M2_M3, M4_limpo e mapeamento numérico (_num).
    2. Cartões: limpeza de categorias raras em card6 e combinação card4 + card6_limpo.
    3. Anomalias no Bloco V: flag is_V14_zero, anomalia em colunas quase-constantes e drop de V305.
    4. Variáveis Temporais, Centavos, Domínios de E-mail e Flags de Nulos/Presença.

    Parâmetros
    ----------
    df_tr : pd.DataFrame
        DataFrame de transações (`df_train_tr`).

    Retorna
    -------
    pd.DataFrame
        DataFrame enriquecido com as novas variáveis.
    """
    # -------------------------------------------------------------------------
    # 1. BLOCO M: FLAGS DE VALIDAÇÃO DE CADASTRO (M1 a M9)
    # -------------------------------------------------------------------------
    cols_m_bool = [c for c in ['M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9'] if c in df_tr.columns]
    if cols_m_bool:
        df_tr['M_true_count'] = (df_tr[cols_m_bool] == 'T').sum(axis=1).astype('int8')
        df_tr['M_false_count'] = (df_tr[cols_m_bool] == 'F').sum(axis=1).astype('int8')

    if all(c in df_tr.columns for c in ['M1', 'M2', 'M3']):
        df_tr['M1_M2_M3_assinatura'] = (
            df_tr['M1'].astype(str).replace('nan', 'X') + '_' +
            df_tr['M2'].astype(str).replace('nan', 'X') + '_' +
            df_tr['M3'].astype(str).replace('nan', 'X')
        )

    if 'M4' in df_tr.columns:
        df_tr['M4_limpo'] = df_tr['M4'].astype(str).replace('nan', 'Missing').astype('category')

    # -------------------------------------------------------------------------
    # 2. COMBINAÇÃO E LIMPEZA DE CARTÃO (card4 e card6)
    # -------------------------------------------------------------------------
    if 'card6' in df_tr.columns:
        df_tr['card6_limpo'] = df_tr['card6'].astype(str).replace({
            'debit or credit': 'debit',
            'charge card': 'credit',
            'nan': 'missing'
        }).fillna('missing')

    if 'card4' in df_tr.columns and 'card6_limpo' in df_tr.columns:
        df_tr['card_brand_type'] = (
            df_tr['card4'].astype(str).replace('nan', 'missing') + '_' + df_tr['card6_limpo'].astype(str)
        ).astype('category')

    # -------------------------------------------------------------------------
    # 3. ANOMALIAS CRÍTICAS E TRATAMENTO NO BLOCO V
    # -------------------------------------------------------------------------
    if 'V14' in df_tr.columns:
        df_tr['is_V14_zero'] = (df_tr['V14'] == 0.0).astype('int8')

    cols_near_const = [c for c in ['V107', 'V117', 'V118', 'V119', 'V120', 'V121', 'V122'] if c in df_tr.columns]
    if cols_near_const:
        df_tr['V_near_constant_anomaly'] = (
            (df_tr[cols_near_const] != 1.0) & df_tr[cols_near_const].notnull()
        ).any(axis=1).astype('int8')

    if 'V305' in df_tr.columns:
        df_tr.drop(columns=['V305'], inplace=True)

    # -------------------------------------------------------------------------
    # 4. MAPEAMENTO NUMÉRICO DO BLOCO M
    # -------------------------------------------------------------------------
    map_tf = {'T': 1, 'F': 0}
    for col in cols_m_bool:
        df_tr[f'{col}_num'] = df_tr[col].map(map_tf)

    # -------------------------------------------------------------------------
    # 5. VARIÁVEIS TEMPORAIS, CENTAVOS, E-MAIL E FLAGS DE PRESENÇA
    # -------------------------------------------------------------------------
    if 'TransactionDT' in df_tr.columns:
        df_tr['Transaction_Hour'] = ((df_tr['TransactionDT'] // 3600) % 24).astype('int8')
        df_tr['Transaction_DayOfWeek'] = ((df_tr['TransactionDT'] // (3600 * 24)) % 7).astype('int8')

    if 'TransactionAmt' in df_tr.columns:
        df_tr['TransactionAmt_Cents'] = np.modf(df_tr['TransactionAmt'])[0].astype('float32')

    if 'P_emaildomain' in df_tr.columns and 'R_emaildomain' in df_tr.columns:
        df_tr['same_email_domain'] = (
            (df_tr['P_emaildomain'] == df_tr['R_emaildomain']) &
            df_tr['P_emaildomain'].notnull()
        ).astype('int8')

    if 'P_emaildomain' in df_tr.columns:
        df_tr['P_email_suffix'] = df_tr['P_emaildomain'].str.split('.').str[-1]

    cols_tr_features = [c for c in df_tr.columns if c not in ['TransactionID', 'isFraud', 'tr_null_count']]
    df_tr['tr_null_count'] = df_tr[cols_tr_features].isnull().sum(axis=1).astype('int16')

    if 'dist2' in df_tr.columns:
        df_tr['has_dist2'] = df_tr['dist2'].notnull().astype('int8')
    if 'R_emaildomain' in df_tr.columns:
        df_tr['has_R_email'] = df_tr['R_emaildomain'].notnull().astype('int8')

    novas_cols = [
        c for c in df_tr.columns
        if any(p in c for p in [
            'M_', 'card_', 'is_V14', 'V_near', '_limpo', '_num',
            'Transaction_Hour', 'Transaction_DayOfWeek', 'TransactionAmt_Cents',
            'same_email_domain', 'P_email_suffix', 'tr_null_count', 'has_dist2', 'has_R_email'
        ])
    ]
    print(f"Feature Engineering concluída em df_train_tr ({len(novas_cols)} features criadas/atualizadas).")
    print(f"Shape atual de df_train_tr: {df_tr.shape}")
    return df_tr


def aplicar_feature_engineering_id(df_id: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica o pipeline de Feature Engineering sobre `df_train_id`:
    1. Extração numérica ordinal de `id_34` (`id_34_score`).
    2. Assinatura e contagem de flags de segurança (`id_35` a `id_38`).
    3. Cruzamento de histórico de dispositivo e IP (`id_28` e `id_29`).
    4. Flag de display padrão vs headless/atípico (`id_32`).
    5. Mapeamento numérico de variáveis binárias/ternárias (`_num`).
    6. Limpeza de família de navegador (`id_31`), área de tela (`id_33`) e fabricante (`DeviceInfo`).

    Parâmetros
    ----------
    df_id : pd.DataFrame
        DataFrame de identidade (`df_train_id`).

    Retorna
    -------
    pd.DataFrame
        DataFrame enriquecido com as novas features de identidade.
    """
    # Garante que o tratamento do bloco esparso já foi aplicado
    if 'id_23' in df_id.columns:
        df_id = tratar_nulos_id(df_id)

    # 1. Extração Numérica Ordinal (id_34)
    if 'id_34' in df_id.columns:
        df_id['id_34_score'] = df_id['id_34'].str.split(':').str[1].astype(float)

    # 2. Assinatura e Contagem de Flags de Segurança (id_35 a id_38)
    flags_seguranca = [c for c in ['id_35', 'id_36', 'id_37', 'id_38'] if c in df_id.columns]
    if len(flags_seguranca) == 4:
        df_id['assinatura_seguranca'] = (
            df_id['id_35'].astype(str) + '_' +
            df_id['id_36'].astype(str) + '_' +
            df_id['id_37'].astype(str) + '_' +
            df_id['id_38'].astype(str)
        )
        df_id['contagem_seguranca'] = (df_id[flags_seguranca] == 'T').sum(axis=1).astype('int8')

    # 3. Cruzamento de Histórico e Confiabilidade (id_28 e id_29)
    if 'id_28' in df_id.columns and 'id_29' in df_id.columns:
        df_id['device_ip_consistency'] = (
            df_id['id_28'].fillna('Miss').astype(str) + '_' + df_id['id_29'].fillna('Miss').astype(str)
        )

    # 4. Flag de Dispositivo Atípico/Bots (id_32 - Profundidade de Cores)
    if 'id_32' in df_id.columns:
        df_id['is_standard_display'] = df_id['id_32'].map(
            lambda x: 1 if x in [24.0, 32.0] else (0 if pd.notnull(x) else -1)
        ).astype('int8')

    # 5. Mapeamento Numérico Direto para Modelagem (Variáveis Binárias/Ternárias)
    binary_maps = {
        'id_12': {'NotFound': 0, 'Found': 1},
        'id_15': {'Unknown': 0, 'New': 1, 'Found': 2},
        'id_16': {'NotFound': 0, 'Found': 1},
        'id_28': {'New': 0, 'Found': 1},
        'id_29': {'NotFound': 0, 'Found': 1},
        'id_35': {'F': 0, 'T': 1},
        'id_36': {'F': 0, 'T': 1},
        'id_37': {'F': 0, 'T': 1},
        'id_38': {'F': 0, 'T': 1},
        'DeviceType': {'desktop': 0, 'mobile': 1}
    }
    for col, mapping in binary_maps.items():
        if col in df_id.columns:
            df_id[f'{col}_num'] = df_id[col].map(mapping)

    # 6. Limpeza de Navegadores (id_31)
    if 'id_31' in df_id.columns:
        df_id['browser_family'] = df_id['id_31'].str.split(' ').str[0]
        df_id['browser_family'] = df_id['browser_family'].str.lower().replace({
            'samsung': 'samsung', 'mobile': 'safari', 'google': 'chrome'
        })

    # 7. Extração Matemática da Resolução (id_33)
    if 'id_33' in df_id.columns:
        df_id['screen_width'] = df_id['id_33'].str.split('x').str[0].astype(float)
        df_id['screen_height'] = df_id['id_33'].str.split('x').str[1].astype(float)
        df_id['screen_area'] = df_id['screen_width'] * df_id['screen_height']

    # 8. Padronização de Fabricantes (DeviceInfo)
    if 'DeviceInfo' in df_id.columns:
        df_id['device_brand'] = 'Other'
        df_id.loc[df_id['DeviceInfo'].str.contains('SM-|SAMSUNG', na=False, case=False), 'device_brand'] = 'Samsung'
        df_id.loc[df_id['DeviceInfo'].str.contains('iOS|Mac|Apple', na=False, case=False), 'device_brand'] = 'Apple'
        df_id.loc[df_id['DeviceInfo'].str.contains('Windows', na=False, case=False), 'device_brand'] = 'Windows'
        df_id.loc[df_id['DeviceInfo'].str.contains('Moto|XT', na=False, case=False), 'device_brand'] = 'Motorola'
        df_id.loc[df_id['DeviceInfo'].str.contains('LG-', na=False, case=False), 'device_brand'] = 'LG'
        df_id.loc[df_id['DeviceInfo'].str.contains('Huawei|ALE-|HUAWEI', na=False, case=False), 'device_brand'] = 'Huawei'

    print(f"Feature Engineering concluída em df_train_id. Shape atual: {df_id.shape}")
    return df_id


def aplicar_ohe(
    df: pd.DataFrame,
    cols: List[str],
    drop_first: bool = True,
) -> pd.DataFrame:
    """
    Aplica One-Hot Encoding às colunas categóricas especificadas,
    substituindo-as pelas dummies no DataFrame.
    """
    cols_existentes = [c for c in cols if c in df.columns]
    if not cols_existentes:
        return df

    df_ohe = pd.get_dummies(
        df[cols_existentes], drop_first=drop_first, dtype='int8'
    )
    df = pd.concat([df.drop(columns=cols_existentes), df_ohe], axis=1)
    print(f"OHE aplicado em {len(cols_existentes)} coluna(s): {cols_existentes}")
    return df


# ------------------------------------------------------------------------------
# 7. INFERÊNCIA E ANÁLISE DO TEST (05_Solucao)
# ------------------------------------------------------------------------------
def aplicar_fe_ieee_cis_base(
    df_tr: pd.DataFrame,
    df_id: pd.DataFrame,
    n_ref: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Replica exatamente a engenharia de atributos determinística do 01_EDA.ipynb
    em tabelas brutas IEEE-CIS (ex.: test_transaction + test_identity).

    Normaliza nomes `id-XX` -> `id_XX` presentes em `test_identity.csv`, preserva
    os campos reais de negócio (`TransactionID`, `TransactionDT`, `amt_usd`, `ProductCD`)
    antes de qualquer downcast ou transformação `log1p`, e ajusta a escala de
    `freq_*` para o tamanho amostral de referência (`n_ref`, quando informado).
    """
    tr = df_tr.copy()
    id_df = df_id.rename(columns=lambda c: c.replace('id-', 'id_') if c.startswith('id-') else c).copy()

    contexto = {
        'TransactionID': tr['TransactionID'].values,
        'TransactionDT': tr['TransactionDT'].values.astype(float),
        'amt_usd': tr['TransactionAmt'].values.astype('float64'),
        'ProductCD': tr['ProductCD'].fillna('Missing').astype(str).values if 'ProductCD' in tr.columns else np.full(len(tr), 'Missing'),
    }

    # Cell 33 (01_EDA): temporais e centavos
    tr['tempo_hora'] = ((tr['TransactionDT'] // 3600) % 24).astype('int8')
    tr['tempo_dia_semana'] = ((tr['TransactionDT'] // (3600 * 24)) % 7).astype('int8')
    tr['tempo_dia_corrido'] = (tr['TransactionDT'] // (3600 * 24)).astype('int16')
    tr['valor_centavos'] = np.modf(tr['TransactionAmt'])[0].astype('float32')

    # Cell 35 (01_EDA): e-mail, nulos em transaction e flags de presença
    tr['email_mesmo_dominio'] = ((tr['P_emaildomain'] == tr['R_emaildomain']) & tr['P_emaildomain'].notnull()).astype('int8')
    tr['email_sufixo_P'] = tr['P_emaildomain'].str.split('.').str[-1]
    cols_tr_features = [c for c in tr.columns if c not in ['TransactionID', 'isFraud']]
    tr['cont_null_tr'] = tr[cols_tr_features].isnull().sum(axis=1).astype('int16')
    tr['flag_tem_dist2'] = tr['dist2'].notnull().astype('int8')
    tr['flag_tem_R_email'] = tr['R_emaildomain'].notnull().astype('int8')

    # Cell 37 (01_EDA): bloco M, cartão e anomalias V
    cols_m_bool = [c for c in ['M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9'] if c in tr.columns]
    tr['match_cont_T'] = (tr[cols_m_bool] == 'T').sum(axis=1).astype('int8')
    tr['match_cont_F'] = (tr[cols_m_bool] == 'F').sum(axis=1).astype('int8')
    tr['match_assinatura_M1M2M3'] = tr['M1'].fillna('X') + '_' + tr['M2'].fillna('X') + '_' + tr['M3'].fillna('X')
    tr['match_M4_cat'] = tr['M4'].fillna('Missing').astype('category')

    tr['cartao_tipo_cat'] = tr['card6'].replace({'debit or credit': 'debit', 'charge card': 'credit'}).fillna('missing')
    tr['cartao_bandeira_tipo'] = (tr['card4'].fillna('missing') + '_' + tr['cartao_tipo_cat']).astype('category')

    if 'V14' in tr.columns:
        tr['flag_V14_zero'] = (tr['V14'] == 0.0).astype('int8')
    cols_near_const = [c for c in ['V107', 'V117', 'V118', 'V119', 'V120', 'V121', 'V122'] if c in tr.columns]
    if cols_near_const:
        tr['flag_V_anomalia_quase_const'] = ((tr[cols_near_const] != 1.0) & tr[cols_near_const].notnull()).any(axis=1).astype('int8')
    if 'V305' in tr.columns:
        tr.drop(columns=['V305'], inplace=True)
    for col in cols_m_bool:
        tr[f'match_{col}_bin'] = tr[col].map({'T': 1, 'F': 0})

    # Cell 39 (01_EDA): renomeação C* -> vel_C* e D* -> delta_D*
    mapa_ren = {f'C{i}': f'vel_C{i}' for i in range(1, 15)}
    mapa_ren.update({f'D{i}': f'delta_D{i}' for i in range(1, 16)})
    tr.rename(columns={k: v for k, v in mapa_ren.items() if k in tr.columns}, inplace=True)

    # Cell 50 (01_EDA): nulos em identity + descarte esparso
    id_df['cont_null_id'] = id_df.isnull().sum(axis=1)
    if 'id_23' in id_df.columns:
        id_df['flag_checagem_rede_avancada'] = id_df['id_23'].notnull().astype('int8')
    id_df.drop(columns=['id_07', 'id_08', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27'],
               inplace=True, errors='ignore')

    # Cell 56 (01_EDA): features de segurança, histórico e dispositivo
    if 'id_34' in id_df.columns:
        id_df['seg_id34_score'] = id_df['id_34'].str.split(':').str[1].astype(float)
    flags_seg = [c for c in ['id_35', 'id_36', 'id_37', 'id_38'] if c in id_df.columns]
    if len(flags_seg) == 4:
        id_df['seg_assinatura_id35_38'] = (
            id_df['id_35'].astype(str) + '_' + id_df['id_36'].astype(str) + '_' +
            id_df['id_37'].astype(str) + '_' + id_df['id_38'].astype(str)
        )
        id_df['seg_cont_T_id35_38'] = (id_df[flags_seg] == 'T').sum(axis=1)
    id_df['hist_disp_ip_consistencia'] = id_df['id_28'].fillna('Miss') + '_' + id_df['id_29'].fillna('Miss')
    id_df['disp_tela_padrao'] = id_df['id_32'].map(
        lambda x: 1 if x in [24.0, 32.0] else (0 if pd.notnull(x) else -1)
    ).astype('int8')

    mapa_ordinal = {
        'id_12': ('hist_id12_bin', {'NotFound': 0, 'Found': 1}),
        'id_15': ('hist_id15_ord', {'Unknown': 0, 'New': 1, 'Found': 2}),
        'id_16': ('hist_id16_bin', {'NotFound': 0, 'Found': 1}),
        'id_28': ('hist_id28_bin', {'New': 0, 'Found': 1}),
        'id_29': ('hist_id29_bin', {'NotFound': 0, 'Found': 1}),
        'id_35': ('seg_id35_bin', {'F': 0, 'T': 1}),
        'id_36': ('seg_id36_bin', {'F': 0, 'T': 1}),
        'id_37': ('seg_id37_bin', {'F': 0, 'T': 1}),
        'id_38': ('seg_id38_bin', {'F': 0, 'T': 1}),
        'DeviceType': ('disp_tipo_bin', {'desktop': 0, 'mobile': 1}),
    }
    for col, (novo_nome, mapping) in mapa_ordinal.items():
        if col in id_df.columns:
            id_df[novo_nome] = id_df[col].map(mapping)

    if 'id_31' in id_df.columns:
        id_df['disp_navegador'] = id_df['id_31'].str.split(' ').str[0].str.lower().replace({
            'samsung': 'samsung', 'mobile': 'safari', 'google': 'chrome'
        })
    if 'id_33' in id_df.columns:
        id_df['disp_tela_largura'] = id_df['id_33'].str.split('x').str[0].astype(float)
        id_df['disp_tela_altura'] = id_df['id_33'].str.split('x').str[1].astype(float)
        id_df['disp_tela_area'] = id_df['disp_tela_largura'] * id_df['disp_tela_altura']
    if 'DeviceInfo' in id_df.columns:
        id_df['disp_marca'] = 'Other'
        for padrao, marca in [
            ('SM-|SAMSUNG', 'Samsung'), ('iOS|Mac|Apple', 'Apple'), ('Windows', 'Windows'),
            ('Moto|XT', 'Motorola'), ('LG-', 'LG'), ('Huawei|ALE-|HUAWEI', 'Huawei'),
        ]:
            id_df.loc[id_df['DeviceInfo'].str.contains(padrao, na=False, case=False), 'disp_marca'] = marca

    # Cell 61 (01_EDA): descarte textual, downcast e merge left
    id_df.drop(columns=['id_31', 'id_33', 'DeviceInfo'], inplace=True, errors='ignore')
    tr = safe_downcast(tr)
    id_df = safe_downcast(id_df)
    df_merged = tr.merge(id_df, on='TransactionID', how='left')

    # Cell 66 (01_EDA): razão card1 e frequency encoding (ajustado p/ escala amostral de referência `n_ref`)
    card1_mean_amt = df_merged.groupby('card1')['TransactionAmt'].transform('mean')
    df_merged['valor_razao_media_card1'] = (df_merged['TransactionAmt'] / (card1_mean_amt + 1e-5)).astype('float32')
    fator_escala = (float(n_ref) / max(len(df_merged), 1)) if n_ref is not None else 1.0
    for col in ['card1', 'card2', 'addr1', 'P_emaildomain']:
        if col in df_merged.columns:
            vc = df_merged[col].value_counts(dropna=False) * fator_escala
            df_merged[f'freq_{col}'] = df_merged[col].map(vc).astype('float32')

    return df_merged, contexto


def preparar_dados_teste_inferencia(
    df_test_tr: pd.DataFrame,
    df_test_id: pd.DataFrame,
    df_master: pd.DataFrame,
    features_modelo: List[str],
    meta: Optional[dict] = None,
    train_ratio: float = 0.60,
    test_ratio: float = 0.20,
) -> Dict[str, Any]:
    """
    Prepara o conjunto cego de teste (Kaggle) com paridade total em relação ao
    `01_EDA.ipynb` + `preparar_dados_modelagem`:
      1. Engenharia de atributos determinística (`aplicar_fe_ieee_cis_base`) com escala amostral `len(df_master)`
      2. Alinhamento `log1p` idêntico ao `df_master` (`_detectar_e_aplicar_log1p_referencia`)
      3. Alinhamento dos mapas `freq_*` com `df_master` (fallback escalado para categorias inéditas)
      4. Estatísticas `card1`/`*_freq` e `OrdinalEncoder` ajustados SOMENTE nos 60% de treino
         de `df_master`, registrando a taxa de categorias desconhecidas (`OOV = -1`).
    """
    from sklearn.preprocessing import OrdinalEncoder

    meta = meta or {}
    cols_log_meta = meta.get('log1p_features') or meta.get('cols_log1p')

    df_fe, contexto = aplicar_fe_ieee_cis_base(df_test_tr, df_test_id, n_ref=len(df_master))
    df_fe, cols_log = _detectar_e_aplicar_log1p_referencia(df_fe, df_master, cols_log_explicitas=cols_log_meta)

    # Alinha `freq_card1`, `freq_card2`, `freq_addr1`, `freq_P_emaildomain` ao mapa de referência do df_master
    # (mantendo o valor escalado e já em log1p/raw de `df_fe` como fallback para categorias inéditas no Test)
    for col in ['card1', 'card2', 'addr1', 'P_emaildomain']:
        f_col = f'freq_{col}'
        if col in df_fe.columns and col in df_master.columns and f_col in df_master.columns:
            mapa_ref = df_master.groupby(col, dropna=False)[f_col].first()
            mapeado = df_fe[col].map(mapa_ref)
            df_fe[f_col] = np.where(mapeado.notna(), mapeado, df_fe[f_col]).astype('float32')

    dados_prep = preparar_dados_modelagem(df_master=df_master, meta=meta, train_ratio=train_ratio, test_ratio=test_ratio)
    train_ref = df_master.sort_values('TransactionDT').iloc[:int(len(df_master) * train_ratio)]
    cat_cols = dados_prep['cat_cols']

    # Mesmo ajuste pós-split de `preparar_dados_modelagem` (razão card1, *_freq e OrdinalEncoder)
    card1_stats = train_ref.groupby('card1')['TransactionAmt'].mean()
    media_global = float(train_ref['TransactionAmt'].mean())
    df_fe['TransactionAmt_to_mean_card1'] = (
        df_fe['TransactionAmt'] / (df_fe['card1'].map(card1_stats).fillna(media_global) + 1e-5)
    ).astype('float32')
    for col in ['card1', 'card2', 'addr1', 'P_emaildomain']:
        if col in df_fe.columns and col in train_ref.columns:
            ref, alvo = train_ref[col], df_fe[col]
            if not pd.api.types.is_numeric_dtype(ref):
                # Paridade de ausentes: o df_master lido do parquet guarda texto ausente como `None`,
                # enquanto o CSV do Test traz `NaN` — chaves diferentes no `map`. Sem normalizar,
                # todo e-mail ausente do Test receberia frequência 0 (no Treino recebe a contagem dos
                # ausentes). Ambos os lados usam o mesmo marcador para ausente.
                ref = ref.astype(object).where(ref.notna(), '__AUSENTE__')
                alvo = alvo.astype(object).where(alvo.notna(), '__AUSENTE__')
            freq_map = ref.value_counts(dropna=False)
            df_fe[f'{col}_freq'] = alvo.map(freq_map).fillna(0).astype('float32')

    registros_oov = []
    cats_ok = [c for c in cat_cols if c in df_fe.columns and c in train_ref.columns]
    if cats_ok:
        ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        ord_enc.fit(train_ref[cats_ok].astype(object).fillna('Missing').astype(str))
        enc_test = ord_enc.transform(df_fe[cats_ok].astype(object).fillna('Missing').astype(str)).astype('int32')
        df_fe[cats_ok] = enc_test
        for idx_c, c in enumerate(cats_ok):
            n_unk_test = int((enc_test[:, idx_c] == -1).sum())
            n_unk_val = int((dados_prep['X_test'][c] == -1).sum()) if c in dados_prep['X_test'].columns else 0
            n_unk_oot = int((dados_prep['X_oot'][c] == -1).sum()) if c in dados_prep['X_oot'].columns else 0
            registros_oov.append({
                'Feature': c,
                'No_Modelo_Final': c in features_modelo,
                'Categorias_Treino': int(len(ord_enc.categories_[idx_c])),
                'OOV_Val_%': n_unk_val / max(len(dados_prep['X_test']), 1) * 100,
                'OOV_OOT_%': n_unk_oot / max(len(dados_prep['X_oot']), 1) * 100,
                'OOV_Test_Qtd': n_unk_test,
                'OOV_Test_%': n_unk_test / max(len(df_fe), 1) * 100,
            })

    df_oov = pd.DataFrame(registros_oov).sort_values(
        ['No_Modelo_Final', 'OOV_Test_%'], ascending=[False, False]
    ).reset_index(drop=True) if registros_oov else pd.DataFrame()

    ausentes = [f for f in features_modelo if f not in df_fe.columns]
    if ausentes:
        raise ValueError(f"Features esperadas pelo modelo ausentes no Test após FE: {ausentes}")

    X_test_final = df_fe[features_modelo].replace([np.inf, -np.inf], np.nan).copy()
    X_in_time = pd.concat(
        [dados_prep['X_train'][features_modelo], dados_prep['X_test'][features_modelo]],
        ignore_index=True,
    )
    X_val = dados_prep['X_test'][features_modelo].copy()
    X_oot = dados_prep['X_oot'][features_modelo].copy()
    val_df = dados_prep['test_df'][['TransactionID', 'TransactionDT', 'TransactionAmt', 'isFraud']].copy()
    oot_df = dados_prep['oot_df'][['TransactionID', 'TransactionDT', 'TransactionAmt', 'isFraud']].copy()

    del df_fe, dados_prep, train_ref
    gc.collect()

    print(f"[Teste] Matriz pronta: {X_test_final.shape[0]:,} linhas × {X_test_final.shape[1]} features | "
          f"{len(cols_log)} colunas em escala log1p.")

    return {
        'X_test': X_test_final,
        'X_in_time': X_in_time,
        'X_val': X_val,
        'X_oot': X_oot,
        'val_df': val_df,
        'oot_df': oot_df,
        'cols_log1p': cols_log,
        'df_oov': df_oov,
        **contexto,
    }


# ==============================================================================
# PIPELINE REPRODUTÍVEL (scripts/01_prepare_data.py)
# ==============================================================================
def aplicar_fe_treino(df_train_tr: pd.DataFrame, df_train_id: pd.DataFrame) -> pd.DataFrame:
    """
    Engenharia de atributos do 01_EDA (Seções 3.2–6.2) sobre as bases de TREINO, usando a
    mesma fonte única da inferência (`aplicar_fe_ieee_cis_base`, sem reescala de `freq_*`).

    Diferença única em relação ao teste: o 01_EDA descarta `tempo_dia_corrido` após o
    painel temporal (Seção 4.0), então ela não segue para o `df_master`.
    """
    df_train, _ = aplicar_fe_ieee_cis_base(df_train_tr, df_train_id, n_ref=None)
    df_train = df_train.drop(columns=['tempo_dia_corrido'], errors='ignore')
    print(f"[FE] Base de treino consolidada (transaction + identity): {df_train.shape}")
    return df_train
