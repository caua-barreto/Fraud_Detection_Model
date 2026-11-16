"""Tema visual corporativo (paleta azul), helpers de título e exportação automática de figuras.

Código migrado de `utils.py` (fachada de compatibilidade dos notebooks).
"""
from __future__ import annotations

from pathlib import Path
from typing import Union, Optional, Dict, Tuple, Any

import os
import re
import sys
import unicodedata

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as _LinearSegmentedColormap


# ==============================================================================
# EXPORTAÇÃO AUTOMÁTICA DE FIGURAS -> reports/figures/ (DIRETÓRIO PLANO)
# ------------------------------------------------------------------------------
# Toda função de gráfico do projeto passa por `_salvar_fig`, `_salvar_figura_opcional`
# ou `_mostrar_figura` (substituto de `plt.show()`), que gravam a figura em um único
# diretório, sem subpastas. O caminho é resolvido a partir da localização deste
# arquivo (não do cwd), então notebooks e scripts exportam para o mesmo lugar.
#
# Controles (sem alterar os notebooks):
#   - variável de ambiente RISCO_EXPORTAR_FIGURAS=0  -> desliga a exportação;
#   - variável de ambiente RISCO_FIGURAS_DIR=<dir>   -> troca o diretório;
#   - configurar_exportacao_figuras(ativo=..., diretorio=...) em código.
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXPORT_CFG: Dict[str, Any] = {
    'ativo': os.environ.get('RISCO_EXPORTAR_FIGURAS', '1').strip().lower() not in ('0', 'false', 'nao', 'não', 'no'),
    'diretorio': Path(os.environ['RISCO_FIGURAS_DIR']) if os.environ.get('RISCO_FIGURAS_DIR')
    else PROJECT_ROOT / 'reports' / 'figures',
    'formato': 'png',
}
_BACKENDS_NAO_INTERATIVOS = {'agg', 'pdf', 'ps', 'svg', 'cairo', 'template', 'pgf'}


def configurar_exportacao_figuras(
    ativo: Optional[bool] = None,
    diretorio: Optional[Union[str, Path]] = None,
    formato: Optional[str] = None,
) -> Dict[str, Any]:
    """Liga/desliga a exportação automática e/ou troca o diretório (único, sem subpastas)."""
    if ativo is not None:
        _EXPORT_CFG['ativo'] = bool(ativo)
    if diretorio is not None:
        _EXPORT_CFG['diretorio'] = Path(diretorio)
    if formato is not None:
        _EXPORT_CFG['formato'] = formato.lstrip('.')
    return dict(_EXPORT_CFG)


def diretorio_figuras() -> Path:
    """Diretório atual de exportação das figuras."""
    return Path(_EXPORT_CFG['diretorio'])


def _slug(texto: str, max_len: int = 70) -> str:
    texto = unicodedata.normalize('NFKD', str(texto)).encode('ascii', 'ignore').decode('ascii')
    texto = re.sub(r'[^A-Za-z0-9]+', '_', texto).strip('_').lower()
    return texto[:max_len].rstrip('_')


def _backend_nao_interativo() -> bool:
    return plt.get_backend().lower() in _BACKENDS_NAO_INTERATIVOS


def _titulo_da_figura(fig: Any) -> str:
    """Extrai o título (suptitle, título dos eixos ou layout Plotly) para compor o nome do arquivo."""
    try:
        if hasattr(fig, 'layout') and not hasattr(fig, 'savefig'):  # Plotly
            return str(fig.layout.title.text or '')
        sup = getattr(fig, '_suptitle', None)
        if sup is not None and sup.get_text():
            return sup.get_text()
        for ax in fig.get_axes():
            for loc in ('left', 'center', 'right'):
                t = ax.get_title(loc=loc)
                if t:
                    return t
    except Exception:
        pass
    return ''


def _funcao_chamadora() -> str:
    """Primeira função pública do pacote `src` na pilha de chamadas (ex.: `plot_roc_pr_janelas`)."""
    frame = sys._getframe(1)
    fallback = 'figura'
    while frame is not None:
        nome = frame.f_code.co_name
        modulo = frame.f_globals.get('__name__', '')
        if modulo != __name__ and modulo.startswith(('src.', 'utils')) and not nome.startswith('_') \
                and nome != '<module>':
            return nome
        if fallback == 'figura' and modulo != __name__ and not nome.startswith('<'):
            fallback = nome.lstrip('_') or 'figura'
        frame = frame.f_back
    return fallback


def _nome_automatico(fig: Any, extensao: str) -> str:
    partes = [_slug(_funcao_chamadora(), 60)]
    titulo = _slug(_titulo_da_figura(fig))
    if titulo:
        partes.append(titulo)
    return '__'.join(p for p in partes if p) + f'.{extensao}'


def _exportar_figura(
    fig: Any,
    nome: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    **savefig_kwargs: Any,
) -> Optional[Path]:
    """Grava `fig` em `diretorio_figuras()/<nome base>`; retorna o caminho (ou None se desativado)."""
    if not _EXPORT_CFG['ativo'] or fig is None:
        return None
    eh_plotly = hasattr(fig, 'write_html') and not hasattr(fig, 'savefig')
    if nome:
        nome_arquivo = Path(str(nome)).name
        if eh_plotly and not nome_arquivo.endswith('.html'):
            nome_arquivo = Path(nome_arquivo).stem + '.html'
    else:
        nome_arquivo = _nome_automatico(fig, 'html' if eh_plotly else _EXPORT_CFG['formato'])
    destino = diretorio_figuras()
    try:
        destino.mkdir(parents=True, exist_ok=True)
        caminho = destino / nome_arquivo
        if eh_plotly:
            fig.write_html(str(caminho), include_plotlyjs='cdn')
        else:
            savefig_kwargs.setdefault('bbox_inches', 'tight')
            fig.savefig(caminho, dpi=dpi, **savefig_kwargs)
            setattr(fig, '_risco_exportada', str(caminho))
        return caminho
    except Exception as exc:  # exportar nunca deve quebrar a análise
        print(f'[figuras] Aviso: não foi possível exportar {nome_arquivo}: {exc}')
        return None


def _mostrar_figura() -> None:
    """
    Substituto de `plt.show()`: exporta as figuras abertas ainda não exportadas para
    `reports/figures/` e exibe. Em backends não interativos (scripts/Agg), fecha as
    figuras após salvar para não acumular memória.
    """
    nomes_usados = set()
    for num in plt.get_fignums():
        fig = plt.figure(num)
        if getattr(fig, '_risco_exportada', None):
            continue
        nome = _nome_automatico(fig, _EXPORT_CFG['formato'])
        base, ext = nome.rsplit('.', 1)
        i = 2
        while nome in nomes_usados:
            nome = f'{base}_{i}.{ext}'
            i += 1
        nomes_usados.add(nome)
        _exportar_figura(fig, nome)
    if _backend_nao_interativo():
        plt.close('all')
    else:
        plt.show()


_PALETA_POLITICAS = ['#97A3B0', '#5DADE2', '#0096C7', '#13579B', '#0B2545']


def _limpar_bordas(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ------------------------------------------------------------------------------
# THRESHOLD FIXO (SEÇÃO 7.2): BUSCA NO TREINO E RESULTADO TOTAL POR BASE
# ------------------------------------------------------------------------------
_CORES_MODELOS = {
    'LGBM_Max':       {'solida': '#0077B6', 'media': '#00B4D8', 'clara': '#90E0EF'},
    'LGBM_Otimizado': {'solida': '#0B2545', 'media': '#13579B', 'clara': '#8DB6E3'},
}


_CORES_EXTRAS = [
    {'solida': '#1D3557', 'media': '#457B9D', 'clara': '#A8DADC'},
    {'solida': '#023E8A', 'media': '#0096C7', 'clara': '#ADE8F4'},
]


def _cor_modelo(nome: str, idx: int = 0) -> Dict[str, str]:
    return _CORES_MODELOS.get(nome, _CORES_EXTRAS[idx % len(_CORES_EXTRAS)])


def _estilo_limpo(ax):
    """Estética padrão dos gráficos da Seção 7: grid horizontal suave, só a borda inferior, sem ticks."""
    ax.grid(axis='y', ls='-', lw=0.6, color='#EAEAEA')
    ax.grid(axis='x', visible=False)
    ax.set_axisbelow(True)
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.tick_params(axis='both', colors='#555555', length=0, pad=5)


def _salvar_figura_opcional(
    fig: Any,
    nome_arquivo: Optional[str] = None,
    save_dir: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> None:
    """
    Exporta a figura (Matplotlib ou Plotly) para `reports/figures/` (diretório plano, sem subpastas).

    - `nome_arquivo` informado -> usa apenas o nome base do arquivo;
    - `nome_arquivo=None` -> nome automático `<funcao>__<titulo>.png|.html`;
    - `save_dir` é aceito por compatibilidade, mas ignorado: todas as figuras vão para o
      diretório único de exportação (`configurar_exportacao_figuras`).
    """
    _exportar_figura(fig, nome_arquivo, dpi=dpi)
    if _backend_nao_interativo() and hasattr(fig, 'savefig'):
        plt.close(fig)


# ==============================================================================
# CONSOLIDAÇÃO DOS ARQUÉTIPOS DE FRAUDE (SEÇÃO 6.0) E BENCHMARK HDBSCAN (SEÇÃO 7.0)
# ==============================================================================
# Paleta categórica validada (todos os pares, inclusive scatter) para os 4 arquétipos.
# Cada cor acompanha a entidade (arquétipo) em todos os gráficos, nunca o ranking.
PALETA_ARQUETIPOS_PADRAO: Tuple[str, ...] = ("#0B2545", "#13579B", "#0096C7", "#5DADE2")


COR_RUIDO_CLUSTER = "#97A3B0"


_COR_TEXTO_PRIMARIO = "#0B2545"


_COR_TEXTO_SECUNDARIO = "#5A6878"


PALETA_AZUL: Dict[str, str] = {
    'marinho': '#0B2545',      # foco principal (OOT, fraude, destaque máximo)
    'principal': '#13579B',
    'medio': '#3E7CC4',
    'ciano': '#0096C7',        # destaque ciano corporativo
    'ciano_claro': '#48CAE4',  # apoio ciano claro
    'claro': '#8DB6E3',        # referência (In-Time)
    'gelo': '#E6F0FA',
    'cinza': '#97A3B0',        # legítimas / contexto
    'cinza_claro': '#D9DEE4',
    'texto': '#1B2633',
    'texto_sec': '#5A6878',
    'grade': '#EBEFF3',
    'acento': '#0096C7',       # destaque ciano para alertas (drift crítico, pior grupo)
}


COR_IN_TIME = PALETA_AZUL['claro']


COR_OOT = PALETA_AZUL['marinho']


COR_LEGITIMA = PALETA_AZUL['cinza']


COR_FRAUDE = PALETA_AZUL['principal']


CMAP_AZUL = _LinearSegmentedColormap.from_list('azul_corporativo', ['#F5F9FD', '#8DB6E3', '#13579B', '#0B2545'])


CMAP_AZUL_SHAP = _LinearSegmentedColormap.from_list('azul_shap', ['#A9C9EC', '#3E7CC4', '#0B2545'])


_CORES_PARTICAO = {'Treino (OOF)': '#FFFFFF', 'Validação': PALETA_AZUL['gelo'], 'OOT': '#D3E3F5'}


def aplicar_estilo_azul() -> None:
    """Aplica o tema visual corporativo (minimalista, paleta azul) em todos os gráficos Matplotlib/Seaborn."""
    from cycler import cycler
    p = PALETA_AZUL
    plt.rcParams.update({
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'figure.dpi': 110, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.edgecolor': p['cinza_claro'], 'axes.linewidth': 1.0,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'axes.axisbelow': True,
        'grid.color': p['grade'], 'grid.linewidth': 0.8, 'grid.linestyle': '-',
        'axes.titlesize': 13, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
        'axes.titlepad': 12, 'axes.titlecolor': p['texto'],
        'axes.labelsize': 10.5, 'axes.labelcolor': p['texto_sec'], 'axes.labelpad': 8,
        'xtick.color': p['texto_sec'], 'ytick.color': p['texto_sec'],
        'xtick.major.size': 0, 'ytick.major.size': 0,
        'legend.frameon': False, 'legend.fontsize': 9.5,
        'axes.prop_cycle': cycler(color=[p['marinho'], p['principal'], p['ciano'], p['claro'], p['cinza']]),
    })


def _titulo_ax(ax, titulo: str, subtitulo: Optional[str] = None) -> None:
    """Título à esquerda em negrito com subtítulo opcional em cinza logo abaixo."""
    ax.set_title(titulo, loc='left', pad=26 if subtitulo else 12)
    if subtitulo:
        ax.text(0, 1.02, subtitulo, transform=ax.transAxes, fontsize=9.5,
                color=PALETA_AZUL['texto_sec'], ha='left', va='bottom')


def _titulo_fig(fig, titulo: str, subtitulo: Optional[str] = None) -> None:
    fig.suptitle(titulo, x=0.01, ha='left', va='bottom', y=1.045 if subtitulo else 1.01,
                 fontsize=15, fontweight='bold', color=PALETA_AZUL['texto'])
    if subtitulo:
        fig.text(0.01, 1.01, subtitulo, ha='left', va='bottom', fontsize=10, color=PALETA_AZUL['texto_sec'])


def _salvar_fig(fig, filepath: Optional[Union[str, Path]] = None) -> None:
    """
    Exporta a figura para `reports/figures/` e exibe.

    `filepath` informado -> usa apenas o nome base (subpastas são achatadas);
    `filepath=None` -> nome automático `<funcao>__<titulo>.png`.
    """
    _exportar_figura(fig, filepath, dpi=300, facecolor='white')
    _mostrar_figura()
