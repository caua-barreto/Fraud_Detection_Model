"""
Leitura, validação e resolução do `configs.yaml` (usado apenas por main.py e scripts/).

- `carregar_config(caminho, overrides)` -> `Config` com acesso por atributo/chave;
- resolução de PROJECT_ROOT a partir deste arquivo (independe do cwd);
- isolamento de experimentos: com `experiment.overwrite_default_artifacts: false`
  todas as SAÍDAS vão para `experiments/<name>/...`, e as ENTRADAS são buscadas
  primeiro no experimento e depois nos caminhos padrão;
- `fixar_seeds(seed)` para reprodutibilidade (random, numpy, PYTHONHASHSEED);
- `registrar_execucao(...)` grava o config resolvido + versões + commit git.

Os notebooks NÃO usam este módulo.
"""
from __future__ import annotations

import copy
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PADRAO = PROJECT_ROOT / 'configs.yaml'

_ESTIMADORES = {'lightgbm', 'xgboost', 'catboost'}
_MODOS_SELECAO = {'all', 'shap_prune', 'custom_list'}
_CALIBRACOES = {'isotonic', 'sigmoid'}
_POLITICAS = {'dynamic_exp', 'fixed'}

# Chave lógica -> (seção em `paths`, subcaminho)
_SAIDAS = {
    'raw': ('raw_dir', None),
    'processed': ('processed_dir', None),
    'artifacts': ('artifacts_dir', None),
    'reports': ('reports_dir', None),
    'figures': ('reports_dir', 'figures'),
    'metrics': ('reports_dir', 'metrics'),
    'submissions': ('submissions_dir', None),
}


class Config(dict):
    """Dicionário com acesso por atributo (cfg.model.tuning.n_trials) e helpers de caminhos."""

    def __getattr__(self, item: str) -> Any:
        try:
            valor = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return Config(valor) if isinstance(valor, dict) and not isinstance(valor, Config) else valor

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    # ------------------------------------------------------------------ caminhos
    @property
    def isolado(self) -> bool:
        return not bool(self['experiment'].get('overwrite_default_artifacts', False))

    @property
    def raiz_saida(self) -> Path:
        """Raiz de saída: PROJECT_ROOT (padrão) ou experiments/<name>/ (experimento isolado)."""
        if not self.isolado:
            return PROJECT_ROOT
        return PROJECT_ROOT / self['paths'].get('experiments_dir', 'experiments') / self['experiment']['name']

    def _rel(self, chave: str) -> Path:
        secao, sub = _SAIDAS[chave]
        rel = Path(self['paths'][secao])
        return rel / sub if sub else rel

    def caminho_padrao(self, chave: str) -> Path:
        return PROJECT_ROOT / self._rel(chave)

    def saida(self, chave: str, criar: bool = True) -> Path:
        """Diretório de saída da etapa (`raw` é sempre o diretório padrão: dados brutos são compartilhados)."""
        destino = self.caminho_padrao(chave) if chave == 'raw' else self.raiz_saida / self._rel(chave)
        if criar:
            destino.mkdir(parents=True, exist_ok=True)
        return destino

    def entrada(self, chave: str, nome_arquivo: str, obrigatorio: bool = True) -> Path:
        """Arquivo de entrada: prioriza a saída do experimento e cai para o caminho padrão."""
        candidatos = [self.saida(chave, criar=False) / nome_arquivo, self.caminho_padrao(chave) / nome_arquivo]
        for c in candidatos:
            if c.exists():
                return c
        if obrigatorio:
            raise FileNotFoundError(
                f"'{nome_arquivo}' não encontrado em: {', '.join(str(c.parent) for c in candidatos)}. "
                "Execute a etapa anterior do pipeline (python main.py --steps ...)."
            )
        return candidatos[0]


# ---------------------------------------------------------------------- leitura
def _merge_profundo(base: Dict[str, Any], novo: Dict[str, Any]) -> Dict[str, Any]:
    saida = copy.deepcopy(base)
    for k, v in (novo or {}).items():
        if isinstance(v, dict) and isinstance(saida.get(k), dict):
            saida[k] = _merge_profundo(saida[k], v)
        else:
            saida[k] = v
    return saida


def _aplicar_override(cfg: Dict[str, Any], expressao: str) -> None:
    """Aplica `a.b.c=valor` (valor interpretado como YAML: números, listas, null, true/false)."""
    if '=' not in expressao:
        raise ValueError(f"Override inválido '{expressao}'. Use chave.subchave=valor.")
    chave, valor = expressao.split('=', 1)
    partes = chave.strip().split('.')
    alvo = cfg
    for p in partes[:-1]:
        alvo = alvo.setdefault(p, {})
        if not isinstance(alvo, dict):
            raise ValueError(f"Override '{expressao}': '{p}' não é uma seção.")
    alvo[partes[-1]] = yaml.safe_load(valor)


def _validar(cfg: Dict[str, Any]) -> None:
    erros: List[str] = []
    d = cfg['data']
    if not (0 < d['train_ratio'] < 1 and 0 < d['val_ratio'] < 1 and d['train_ratio'] + d['val_ratio'] < 1):
        erros.append('data.train_ratio e data.val_ratio devem ser > 0 e somar < 1 (o restante é o OOT).')
    m = cfg['model']
    if m['estimator'] not in _ESTIMADORES:
        erros.append(f"model.estimator deve ser um de {sorted(_ESTIMADORES)}.")
    fs = m['feature_selection']
    if fs['mode'] not in _MODOS_SELECAO:
        erros.append(f"model.feature_selection.mode deve ser um de {sorted(_MODOS_SELECAO)}.")
    if fs['mode'] == 'custom_list' and not fs.get('custom_list'):
        erros.append('model.feature_selection.custom_list não pode ser vazio com mode: custom_list.')
    if fs.get('select', 'otimizado') not in ('otimizado', 'max'):
        erros.append("model.feature_selection.select deve ser 'otimizado' ou 'max'.")
    if not m['tuning']['run_optuna'] and not m['tuning'].get('params_override'):
        erros.append('model.tuning.params_override é obrigatório com run_optuna: false.')
    if cfg['calibration']['method'] not in _CALIBRACOES:
        erros.append(f"calibration.method deve ser um de {sorted(_CALIBRACOES)}.")
    dp = cfg['decision_policy']
    if dp['mode'] not in _POLITICAS:
        erros.append(f"decision_policy.mode deve ser um de {sorted(_POLITICAS)}.")
    if dp['mdr_rate'] < 0 or dp['fraud_loss_rate'] <= 0:
        erros.append('decision_policy.mdr_rate deve ser >= 0 e fraud_loss_rate > 0.')
    if not str(cfg['experiment'].get('name', '')).strip():
        erros.append('experiment.name não pode ser vazio.')
    if erros:
        raise ValueError('configs.yaml inválido:\n  - ' + '\n  - '.join(erros))


def carregar_config(
    caminho: Optional[Union[str, Path]] = None,
    overrides: Optional[Iterable[str]] = None,
) -> Config:
    """Lê o YAML, aplica overrides `chave=valor` e valida."""
    caminho = Path(caminho) if caminho else CONFIG_PADRAO
    if not caminho.is_absolute() and not caminho.exists():
        caminho = PROJECT_ROOT / caminho
    with open(caminho, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    for exp in overrides or []:
        _aplicar_override(cfg, exp)
    _validar(cfg)
    cfg = Config(cfg)
    dict.__setitem__(cfg, '_arquivo', str(caminho))
    return cfg


# --------------------------------------------------------------- reprodutibilidade
def fixar_seeds(seed: int) -> None:
    """Fixa as fontes de aleatoriedade globais (os estimadores recebem `random_state` explícito)."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def _commit_git() -> Optional[str]:
    try:
        r = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=PROJECT_ROOT,
                           capture_output=True, text=True, timeout=5)
        sujo = subprocess.run(['git', 'status', '--porcelain'], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, timeout=5).stdout.strip()
        return (r.stdout.strip() + ('-dirty' if sujo else '')) if r.returncode == 0 else None
    except Exception:
        return None


def versoes_pacotes() -> Dict[str, Optional[str]]:
    from importlib import metadata
    pacotes = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'xgboost', 'catboost', 'optuna',
               'shap', 'umap-learn', 'matplotlib', 'pyarrow', 'joblib', 'pyyaml']
    saida = {}
    for p in pacotes:
        try:
            saida[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            saida[p] = None
    return saida


def registrar_execucao(cfg: Config, etapas: List[str], status: Dict[str, Any]) -> Path:
    """Grava `run_manifest.json` (config resolvido, etapas, status, versões, commit) e `configs_resolvido.yaml`."""
    raiz = cfg.raiz_saida
    raiz.mkdir(parents=True, exist_ok=True)
    cfg_limpo = {k: v for k, v in cfg.items() if not k.startswith('_')}
    with open(raiz / 'configs_resolvido.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg_limpo, f, allow_unicode=True, sort_keys=False)
    manifesto = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'experimento': cfg['experiment']['name'],
        'isolado': cfg.isolado,
        'raiz_saida': str(raiz),
        'config_origem': cfg.get('_arquivo'),
        'etapas': etapas,
        'status': status,
        'git_commit': _commit_git(),
        'python': sys.version.split()[0],
        'plataforma': platform.platform(),
        'pacotes': versoes_pacotes(),
    }
    caminho = raiz / 'run_manifest.json'
    with open(caminho, 'w', encoding='utf-8') as f:
        json.dump(manifesto, f, indent=2, ensure_ascii=False, default=str)
    return caminho


def preparar_ambiente_execucao(cfg: Config) -> None:
    """Backend gráfico headless, seeds e diretório de figuras do experimento (chamado pelos scripts)."""
    import matplotlib
    matplotlib.use('Agg', force=True)
    import warnings
    warnings.filterwarnings('ignore')
    fixar_seeds(int(cfg['experiment']['random_state']))
    from src.visualization.theme import aplicar_estilo_azul, configurar_exportacao_figuras
    aplicar_estilo_azul()
    configurar_exportacao_figuras(ativo=True, diretorio=cfg.saida('figures'))


# ------------------------------------------------------------------------- I/O
def para_json(obj: Any) -> Any:
    """Converte tipos numpy/pandas/Path em tipos nativos JSON (recursivo)."""
    import pandas as pd

    if isinstance(obj, dict):
        return {str(k): para_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [para_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return para_json(obj.tolist())
    if isinstance(obj, (pd.Series,)):
        return para_json(obj.to_dict())
    if isinstance(obj, (pd.DataFrame,)):
        return para_json(obj.to_dict(orient='records'))
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def salvar_json(obj: Any, caminho: Union[str, Path]) -> Path:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, 'w', encoding='utf-8') as f:
        json.dump(para_json(obj), f, indent=2, ensure_ascii=False)
    return caminho


# ------------------------------------------------------------------------- CLI
def parser_padrao(descricao: str):
    import argparse

    p = argparse.ArgumentParser(description=descricao)
    p.add_argument('--config', default=str(CONFIG_PADRAO), help='Caminho do YAML (padrão: configs.yaml).')
    p.add_argument('--set', dest='overrides', action='append', default=[], metavar='CHAVE=VALOR',
                   help='Sobrescreve uma chave do YAML (pode repetir). Ex.: --set model.tuning.n_trials=30')
    p.add_argument('--experiment', default=None, help='Atalho para --set experiment.name=<nome>.')
    p.add_argument('--overwrite-default', action='store_true',
                   help='Grava nos caminhos padrão (equivale a experiment.overwrite_default_artifacts=true).')
    return p


def config_de_args(args) -> Config:
    overrides = list(args.overrides or [])
    if getattr(args, 'experiment', None):
        overrides.append(f'experiment.name={args.experiment}')
    if getattr(args, 'overwrite_default', False):
        overrides.append('experiment.overwrite_default_artifacts=true')
    return carregar_config(args.config, overrides)


def executar_script(nome_etapa: str, descricao: str, run, argv: Optional[List[str]] = None) -> Any:
    """Ponto de entrada padrão dos scripts/ quando executados diretamente pelo terminal."""
    args = parser_padrao(descricao).parse_args(argv)
    cfg = config_de_args(args)
    preparar_ambiente_execucao(cfg)
    print(f"[{nome_etapa}] experimento='{cfg['experiment']['name']}' | saída: {cfg.raiz_saida}")
    resultado = run(cfg)
    registrar_execucao(cfg, [nome_etapa], {nome_etapa: 'ok'})
    return resultado
