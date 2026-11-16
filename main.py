"""
main.py — Orquestrador CLI único do pipeline reprodutível (IEEE-CIS Fraud Detection)

Executa, apenas com Python (sem notebooks), as etapas em scripts/:

    prepare   -> scripts/01_prepare_data.py   (raw -> df_master.parquet + features_metadata.json)
    train     -> scripts/02_train.py          (df_master -> artefato .joblib + metadata/metrics)
    segment   -> scripts/05_segment_fraud.py  (opcional: clusters/arquétipos das fraudes -> cache .parquet)
    evaluate  -> scripts/03_evaluate_oot.py   (In-Time x OOT, drift, SHAP -> monitoramento_oot.json)
    predict   -> scripts/04_predict_test.py   (test cego -> submission_kaggle.csv)

Exemplos:
    python main.py                                   # pipeline completo com configs.yaml
    python main.py --steps train evaluate            # só algumas etapas
    python main.py --experiment xgb_fixo \\
        --set model.estimator=xgboost --set decision_policy.mode=fixed
    python main.py --set model.tuning.run_optuna=false --steps train
    python main.py --overwrite-default               # grava nos caminhos usados pelos notebooks
    python main.py --list                            # mostra etapas e saída resolvida

Com `experiment.overwrite_default_artifacts: false` (padrão), tudo é gravado em
experiments/<name>/ e os arquivos lidos pelos notebooks não são alterados.
"""
from __future__ import annotations

import importlib.util
import sys
import time
import traceback
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.config import config_de_args, parser_padrao, preparar_ambiente_execucao, registrar_execucao  # noqa: E402

ETAPAS = {
    'prepare': 'scripts/01_prepare_data.py',
    'train': 'scripts/02_train.py',
    'segment': 'scripts/05_segment_fraud.py',
    'evaluate': 'scripts/03_evaluate_oot.py',
    'predict': 'scripts/04_predict_test.py',
}
ORDEM_PADRAO = list(ETAPAS)


def carregar_script(caminho_relativo: str):
    """Importa um script de scripts/ (nomes começam com dígito, então não são importáveis via `import`)."""
    caminho = RAIZ / caminho_relativo
    nome = 'pipeline_' + caminho.stem
    spec = importlib.util.spec_from_file_location(nome, caminho)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[nome] = modulo
    spec.loader.exec_module(modulo)
    return modulo


def main(argv=None) -> int:
    parser = parser_padrao(__doc__.strip().splitlines()[1])
    parser.add_argument('--steps', nargs='+', choices=ORDEM_PADRAO, default=None,
                        help=f"Etapas a executar (padrão: todas, na ordem {' -> '.join(ORDEM_PADRAO)}).")
    parser.add_argument('--skip', nargs='+', choices=ORDEM_PADRAO, default=[], help='Etapas a pular.')
    parser.add_argument('--continue-on-error', action='store_true', help='Não interrompe o pipeline se uma etapa falhar.')
    parser.add_argument('--list', action='store_true', help='Lista as etapas e a saída resolvida, sem executar.')
    args = parser.parse_args(argv)

    cfg = config_de_args(args)
    pedidas = set(args.steps or ORDEM_PADRAO)
    etapas = [e for e in ORDEM_PADRAO if e in pedidas and e not in set(args.skip)]
    if not cfg['segmentation']['enabled'] and 'segment' in etapas and not args.steps:
        etapas.remove('segment')

    print('=' * 78)
    print(f" Experimento : {cfg['experiment']['name']} ({'isolado' if cfg.isolado else 'caminhos padrão'})")
    print(f" Saída       : {cfg.raiz_saida}")
    print(f" Etapas      : {' -> '.join(etapas)}")
    print('=' * 78)
    if args.list:
        for e in ORDEM_PADRAO:
            print(f"  {'[x]' if e in etapas else '[ ]'} {e:<9} {ETAPAS[e]}")
        return 0

    preparar_ambiente_execucao(cfg)
    status = {}
    for etapa in etapas:
        t0 = time.time()
        print(f"\n{'#' * 78}\n# ETAPA: {etapa} ({ETAPAS[etapa]})\n{'#' * 78}")
        try:
            resultado = carregar_script(ETAPAS[etapa]).run(cfg)
            status[etapa] = {'status': 'ok', 'segundos': round(time.time() - t0, 1), 'saidas': resultado}
        except Exception as exc:  # registra a falha no manifesto antes de decidir se segue
            traceback.print_exc()
            status[etapa] = {'status': 'erro', 'segundos': round(time.time() - t0, 1), 'erro': repr(exc)}
            if not args.continue_on_error:
                break
        finally:
            registrar_execucao(cfg, etapas, status)

    print('\n' + '=' * 78)
    for etapa, s in status.items():
        print(f" {etapa:<9} {s['status']:<5} {s['segundos']:>8.1f}s")
    print(f" Manifesto: {cfg.raiz_saida / 'run_manifest.json'}")
    print('=' * 78)
    return 0 if all(s['status'] == 'ok' for s in status.values()) and len(status) == len(etapas) else 1


if __name__ == '__main__':
    sys.exit(main())
