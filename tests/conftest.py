"""Fixtures com dados sintéticos no formato bruto do IEEE-CIS (sem depender do Kaggle)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))


def _bases_brutas(n: int, seed: int, id_inicio: int, dt_inicio: int, com_alvo: bool, hifen: bool):
    rng = np.random.default_rng(seed)
    ids = np.arange(id_inicio, id_inicio + n)
    tf = lambda: rng.choice(['T', 'F', None], size=n)  # noqa: E731
    tr = pd.DataFrame({
        'TransactionID': ids,
        'TransactionDT': np.sort(rng.integers(dt_inicio, dt_inicio + 86400 * 180, size=n)),
        'TransactionAmt': np.round(rng.lognormal(4, 1, size=n), 3),
        'ProductCD': rng.choice(['W', 'C', 'H', 'R', 'S'], size=n),
        'card1': rng.integers(1000, 1100, size=n),
        'card2': rng.choice([100.0, 200.0, np.nan], size=n),
        'card4': rng.choice(['visa', 'mastercard', None], size=n),
        'card5': rng.choice([150.0, 226.0], size=n),
        'card6': rng.choice(['debit', 'credit', 'debit or credit', 'charge card', None], size=n),
        'addr1': rng.choice([299.0, 325.0, np.nan], size=n),
        'addr2': rng.choice([87.0, np.nan], size=n),
        'dist2': rng.choice([1.0, np.nan], size=n),
        'P_emaildomain': rng.choice(['gmail.com', 'yahoo.com.mx', None], size=n),
        'R_emaildomain': rng.choice(['gmail.com', None], size=n),
    })
    for i in range(1, 10):
        tr[f'M{i}'] = tf() if i != 4 else rng.choice(['M0', 'M1', None], size=n)
    for i in range(1, 15):
        tr[f'C{i}'] = rng.poisson(2, size=n).astype(float)
    for i in range(1, 16):
        tr[f'D{i}'] = np.where(rng.random(n) < 0.3, np.nan, rng.integers(0, 500, size=n)).astype(float)
    for v in ['V14', 'V107', 'V117', 'V118', 'V119', 'V120', 'V121', 'V122', 'V305', 'V70', 'V258']:
        tr[v] = rng.choice([0.0, 1.0, np.nan], size=n)
    if com_alvo:
        tr.insert(1, 'isFraud', (rng.random(n) < 0.1).astype(int))

    n_id = n // 2
    sep = '-' if hifen else '_'
    idf = pd.DataFrame({'TransactionID': rng.choice(ids, size=n_id, replace=False)})
    for i in [1, 2, 5, 6, 7, 8, 11, 13, 14, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 32]:
        idf[f'id{sep}{i:02d}'] = rng.choice([24.0, 32.0, 16.0, np.nan], size=n_id)
    idf[f'id{sep}12'] = rng.choice(['Found', 'NotFound'], size=n_id)
    idf[f'id{sep}15'] = rng.choice(['Found', 'New', 'Unknown', None], size=n_id)
    idf[f'id{sep}16'] = rng.choice(['Found', 'NotFound', None], size=n_id)
    idf[f'id{sep}28'] = rng.choice(['Found', 'New', None], size=n_id)
    idf[f'id{sep}29'] = rng.choice(['Found', 'NotFound', None], size=n_id)
    idf[f'id{sep}31'] = rng.choice(['chrome 63.0', 'mobile safari 11.0', 'samsung browser', None], size=n_id)
    idf[f'id{sep}33'] = rng.choice(['1920x1080', '1334x750', None], size=n_id)
    idf[f'id{sep}34'] = rng.choice(['match_status:2', 'match_status:1', None], size=n_id)
    for i in [35, 36, 37, 38]:
        idf[f'id{sep}{i}'] = rng.choice(['T', 'F', None], size=n_id)
    idf['DeviceType'] = rng.choice(['desktop', 'mobile', None], size=n_id)
    idf['DeviceInfo'] = rng.choice(['Windows', 'iOS Device', 'SM-G920', 'Moto G', None], size=n_id)
    return tr, idf


@pytest.fixture(scope='session')
def bases_treino():
    return _bases_brutas(4000, seed=1, id_inicio=2_987_000, dt_inicio=86400, com_alvo=True, hifen=False)


@pytest.fixture(scope='session')
def bases_teste():
    # test_identity.csv do Kaggle usa hífen nos nomes (id-01 ... id-38)
    return _bases_brutas(1500, seed=2, id_inicio=3_663_549, dt_inicio=86400 * 213, com_alvo=False, hifen=True)
