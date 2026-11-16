"""
Gera o relatório de negócio em PDF (Relatorio_Analise_Fraude.pdf, na raiz do projeto).

O conteúdo narrativo usa os resultados consolidados dos notebooks 01–05 e as figuras
exportadas em reports/figures/. Para regerar após editar o texto:

    pip install reportlab==4.2.5
    python scripts/gerar_relatorio_pdf.py
"""
from __future__ import annotations

import shutil
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether, NextPageTemplate, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)

RAIZ = Path(__file__).resolve().parents[1]
FIG = RAIZ / 'reports' / 'figures'
SAIDA = RAIZ / 'Relatorio_Analise_Fraude.pdf'

# ------------------------------------------------------------------------------ identidade visual
MARINHO = colors.HexColor('#0B2545')
PRINCIPAL = colors.HexColor('#13579B')
MEDIO = colors.HexColor('#3E7CC4')
CIANO = colors.HexColor('#0096C7')
CLARO = colors.HexColor('#8DB6E3')
GELO = colors.HexColor('#E6F0FA')
TEXTO = colors.HexColor('#1B2633')
CINZA = colors.HexColor('#5A6878')
LINHA = colors.HexColor('#D9DEE4')
BRANCO = colors.white

FONTES = Path('/usr/share/fonts/truetype/lato')   # Lato (Linux: apt install fonts-lato); senão, Helvetica
_ARQS = {'R': 'Lato-Regular.ttf', 'B': 'Lato-Bold.ttf', 'I': 'Lato-Italic.ttf', 'L': 'Lato-Light.ttf',
         'K': 'Lato-Black.ttf', 'S': 'Lato-Semibold.ttf'}
if all((FONTES / a).exists() for a in _ARQS.values()):
    F = {k: Path(a).stem for k, a in _ARQS.items()}
    for k, a in _ARQS.items():
        pdfmetrics.registerFont(TTFont(F[k], str(FONTES / a)))
    pdfmetrics.registerFontFamily(F['R'], normal=F['R'], bold=F['B'], italic=F['I'], boldItalic=F['B'])
else:
    F = {'R': 'Helvetica', 'B': 'Helvetica-Bold', 'I': 'Helvetica-Oblique', 'L': 'Helvetica',
         'K': 'Helvetica-Bold', 'S': 'Helvetica-Bold'}

# ------------------------------------------------------------------------------ dados do autor
AUTOR = 'Cauã de Freitas Barreto'
CARGO = 'Junior Data Scientist'
GITHUB = 'https://github.com/caua-barreto'
LINKEDIN = 'https://www.linkedin.com/in/caua-fbarreto-'
REPOSITORIO = 'https://github.com/caua-barreto/Fraud_Detection_Model'

LARG, ALT = A4
MARG = 2.0 * cm
UTIL = LARG - 2 * MARG

ST = {
    'corpo': ParagraphStyle('corpo', fontName=F['R'], fontSize=10.2, leading=15, textColor=TEXTO, alignment=TA_JUSTIFY,
                            spaceAfter=7),
    'h1': ParagraphStyle('h1', fontName=F['K'], fontSize=20, leading=24, textColor=MARINHO, spaceBefore=4,
                         spaceAfter=4),
    'kicker': ParagraphStyle('kicker', fontName=F['B'], fontSize=8.5, leading=11, textColor=CIANO, spaceAfter=2),
    'lead': ParagraphStyle('lead', fontName=F['L'], fontSize=12, leading=17, textColor=CINZA, spaceAfter=12),
    'h2': ParagraphStyle('h2', fontName=F['B'], fontSize=13, leading=17, textColor=PRINCIPAL, spaceBefore=10,
                         spaceAfter=5),
    'lista': ParagraphStyle('lista', fontName=F['R'], fontSize=10.2, leading=15, textColor=TEXTO, leftIndent=14,
                            bulletIndent=3, spaceAfter=4, alignment=TA_LEFT),
    'legenda': ParagraphStyle('legenda', fontName=F['I'], fontSize=8.5, leading=11, textColor=CINZA,
                              alignment=TA_CENTER, spaceBefore=3, spaceAfter=10),
    'callout': ParagraphStyle('callout', fontName=F['R'], fontSize=10, leading=14.5, textColor=MARINHO),
    'tab_h': ParagraphStyle('tab_h', fontName=F['B'], fontSize=8.6, leading=11, textColor=BRANCO, alignment=TA_CENTER),
    'tab': ParagraphStyle('tab', fontName=F['R'], fontSize=8.6, leading=11, textColor=TEXTO, alignment=TA_CENTER),
    'tab_l': ParagraphStyle('tab_l', fontName=F['R'], fontSize=8.6, leading=11, textColor=TEXTO, alignment=TA_LEFT),
    'kpi_v': ParagraphStyle('kpi_v', fontName=F['K'], fontSize=14.5, leading=18, textColor=MARINHO, alignment=TA_CENTER),
    'kpi_r': ParagraphStyle('kpi_r', fontName=F['R'], fontSize=7.8, leading=10, textColor=CINZA, alignment=TA_CENTER),
}


# ------------------------------------------------------------------------------ componentes
def p(texto, estilo='corpo'):
    return Paragraph(texto, ST[estilo])


def bullets(itens):
    return [Paragraph(t, ST['lista'], bulletText='•') for t in itens]


def secao(numero, titulo, lead=None):
    blocos = [p(f'CAPÍTULO {numero}', 'kicker') if numero else Spacer(1, 0), p(titulo, 'h1')]
    if lead:
        blocos.append(p(lead, 'lead'))
    return blocos


# ------------------------------------------------------------------------------ figuras
# Os PNGs dos notebooks foram desenhados largos (15–18 polegadas). Para o texto interno não
# ficar minúsculo no A4: (1) as bordas brancas são cortadas, (2) figuras com vários painéis
# são separadas pelas faixas brancas entre eles e (3) os painéis são rearranjados em grade.
# As cópias processadas ficam em um diretório temporário; reports/figures/ não é alterado.
_TMP = Path(tempfile.mkdtemp(prefix='relatorio_figs_'))
_BRANCO = 240


def _cortar_bordas(arr, folga=10):
    conteudo = (arr < _BRANCO).any(axis=2)
    linhas, colunas = np.where(conteudo.any(axis=1))[0], np.where(conteudo.any(axis=0))[0]
    if not len(linhas) or not len(colunas):
        return arr
    y0, y1 = max(linhas[0] - folga, 0), min(linhas[-1] + folga + 1, arr.shape[0])
    x0, x1 = max(colunas[0] - folga, 0), min(colunas[-1] + folga + 1, arr.shape[1])
    return arr[y0:y1, x0:x1]


def _paineis(arr, min_faixa=12, juntar=40, min_frac=0.15):
    """Separa painéis lado a lado usando colunas inteiramente brancas como divisórias."""
    w = arr.shape[1]
    branco = (arr >= _BRANCO).all(axis=2).all(axis=0)
    faixas, i = [], 0
    while i < w:
        if branco[i]:
            j = i
            while j < w and branco[j]:
                j += 1
            if j - i >= min_faixa and 0.05 * w < (i + j) / 2 < 0.95 * w:   # ignora as margens externas
                faixas.append((i, j))
            i = j
        else:
            i += 1
    # Faixas próximas formam um grupo (ex.: o rótulo vertical do eixo Y do painel seguinte fica entre
    # duas faixas brancas). O corte é feito na PRIMEIRA faixa do grupo, para o rótulo ficar inteiro
    # junto do painel a que pertence.
    cortes, fim_anterior = [], None
    for a, b in faixas:
        if fim_anterior is None or a - fim_anterior > juntar:
            cortes.append((a + b) // 2)
        fim_anterior = b
    limites = [0] + cortes + [w]
    return [arr[:, a:b] for a, b in zip(limites[:-1], limites[1:]) if b - a >= min_frac * w]


def _preparar(nome, painel=None):
    """Caminho de uma cópia processada da figura (bordas cortadas e, opcionalmente, um único painel)."""
    arr = np.asarray(PILImage.open(FIG / f'{nome}.png').convert('RGB'))
    if painel is not None:
        arr = _paineis(arr)[painel]
    arr = _cortar_bordas(arr)
    destino = _TMP / f'{nome}{"" if painel is None else f"_p{painel}"}.png'
    PILImage.fromarray(arr).save(destino)
    return destino


def _imagem(caminho, largura, max_alt):
    w, h = ImageReader(str(caminho)).getSize()
    larg, alt = largura, largura * h / w
    if alt > max_alt:
        alt, larg = max_alt, max_alt * w / h
    img = Image(str(caminho), width=larg, height=alt)
    img.hAlign = 'CENTER'
    return img


def figura(nome, legenda, largura=UTIL, max_alt=9.0 * cm, painel=None):
    return KeepTogether([_imagem(_preparar(nome, painel), largura, max_alt), p(legenda, 'legenda')])


def figura_grade(itens, legenda, max_alt=8.5 * cm, espaco=0.4 * cm):
    """Painéis lado a lado ocupando a largura útil: itens = [(nome, painel), ...]."""
    larg = (UTIL - espaco * (len(itens) - 1)) / len(itens)
    imgs = [_imagem(_preparar(nome, pn), larg, max_alt) for nome, pn in itens]
    t = Table([imgs], colWidths=[larg + (espaco if i < len(itens) - 1 else 0) for i in range(len(itens))])
    t.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'BOTTOM'), ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                           ('LEFTPADDING', (0, 0), (-1, -1), 0), ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                           ('TOPPADDING', (0, 0), (-1, -1), 0), ('BOTTOMPADDING', (0, 0), (-1, -1), 0)]))
    return KeepTogether([t, p(legenda, 'legenda')])


def callout(titulo, texto, cor=CIANO):
    conteudo = Paragraph(f'<font name="{F['B']}" color="#{cor.hexval()[2:]}">{titulo}</font><br/>{texto}',
                         ST['callout'])
    t = Table([[conteudo]], colWidths=[UTIL])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), GELO),
        ('LINEBEFORE', (0, 0), (0, -1), 3.5, cor),
        ('LEFTPADDING', (0, 0), (-1, -1), 12), ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 9), ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    return KeepTogether([Spacer(1, 4), t, Spacer(1, 10)])


def tabela(cabecalho, linhas, larguras=None, destaque=None, primeira_esq=True, esq=()):
    dados = [[Paragraph(c, ST['tab_h']) for c in cabecalho]]
    for linha in linhas:
        dados.append([Paragraph(str(c), ST['tab_l'] if ((j == 0 and primeira_esq) or j in esq) else ST['tab'])
                      for j, c in enumerate(linha)])
    if larguras is None:
        larguras = [UTIL / len(cabecalho)] * len(cabecalho)
    else:
        total = sum(larguras)
        larguras = [UTIL * x / total for x in larguras]
    t = Table(dados, colWidths=larguras, repeatRows=1)
    estilo = [
        ('BACKGROUND', (0, 0), (-1, 0), MARINHO),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LINEBELOW', (0, 1), (-1, -1), 0.5, LINHA),
        ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6), ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]
    for i in range(1, len(dados)):
        if i % 2 == 0:
            estilo.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#F5F9FD')))
    for i in destaque or []:
        estilo += [('BACKGROUND', (0, i + 1), (-1, i + 1), GELO),
                   ('FONTNAME', (0, i + 1), (-1, i + 1), F['B'])]
    t.setStyle(TableStyle(estilo))
    return KeepTogether([t, Spacer(1, 10)])


def kpis(itens):
    """Cartões de indicador: [(valor, rótulo, cor_da_barra), ...]."""
    celulas = []
    for valor, rotulo, cor in itens:
        celulas.append([Paragraph(valor, ST['kpi_v']), Paragraph(rotulo, ST['kpi_r'])])
    linha = [Table([[c[0]], [c[1]]], colWidths=[UTIL / len(itens) - 8]) for c in celulas]
    for t, (_, _, cor) in zip(linha, itens):
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), GELO), ('LINEABOVE', (0, 0), (-1, 0), 3, cor),
            ('TOPPADDING', (0, 0), (-1, -1), 7), ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ]))
    grade = Table([linha], colWidths=[UTIL / len(itens)] * len(itens))
    grade.setStyle(TableStyle([('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 4)]))
    return KeepTogether([grade, Spacer(1, 12)])


def cartao_autor():
    """Cartão final com os dados de contato do autor (links clicáveis)."""
    def link(url):
        return f'<link href="{url}" color="#13579B"><u>{url.replace("https://", "")}</u></link>'

    st_nome = ParagraphStyle('a_nome', fontName=F['K'], fontSize=15, leading=19, textColor=BRANCO)
    st_cargo = ParagraphStyle('a_cargo', fontName=F['L'], fontSize=10.5, leading=14, textColor=colors.HexColor('#CFE0F3'))
    st_rot = ParagraphStyle('a_rot', fontName=F['B'], fontSize=8, leading=11, textColor=CIANO)
    st_val = ParagraphStyle('a_val', fontName=F['R'], fontSize=8.8, leading=12, textColor=TEXTO)

    esquerda = [Paragraph('SOBRE O AUTOR', ParagraphStyle('a_k', fontName=F['B'], fontSize=8, leading=11,
                                                          textColor=CLARO, spaceAfter=6)),
                Paragraph(AUTOR, st_nome), Spacer(1, 3), Paragraph(CARGO, st_cargo)]
    linhas = [[Paragraph('GITHUB', st_rot), Paragraph(link(GITHUB), st_val)]]
    if LINKEDIN:
        linhas.append([Paragraph('LINKEDIN', st_rot), Paragraph(link(LINKEDIN), st_val)])
    linhas.append([Paragraph('PROJETO', st_rot), Paragraph(link(REPOSITORIO), st_val)])
    direita = Table(linhas, colWidths=[2.0 * cm, UTIL * 0.62 - 2.0 * cm - 26])
    direita.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                 ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                                 ('LEFTPADDING', (0, 0), (-1, -1), 0), ('LINEBELOW', (0, 0), (-1, -2), 0.5, LINHA)]))
    t = Table([[esquerda, direita]], colWidths=[UTIL * 0.38, UTIL * 0.62])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), MARINHO), ('BACKGROUND', (1, 0), (1, 0), GELO),
        ('LINEABOVE', (0, 0), (-1, 0), 3, CIANO), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 16), ('RIGHTPADDING', (0, 0), (-1, -1), 14),
        ('TOPPADDING', (0, 0), (-1, -1), 16), ('BOTTOMPADDING', (0, 0), (-1, -1), 16),
    ]))
    return KeepTogether([t])


# ------------------------------------------------------------------------------ páginas
def capa(c, doc):
    c.saveState()
    c.setFillColor(MARINHO)
    c.rect(0, 0, LARG, ALT, stroke=0, fill=1)
    c.setFillColor(PRINCIPAL)
    c.rect(0, 0, LARG, ALT * 0.36, stroke=0, fill=1)
    c.setFillColor(CIANO)
    c.rect(MARG, ALT * 0.36 - 3, 4.2 * cm, 6, stroke=0, fill=1)
    # padrão discreto de pontos (identidade)
    c.setFillColor(colors.HexColor('#1B3A5C'))
    for i in range(14):
        for j in range(6):
            c.circle(LARG - MARG - i * 0.55 * cm, ALT - 2.2 * cm - j * 0.55 * cm, 1.4, stroke=0, fill=1)
    c.setFillColor(CLARO)
    c.setFont(F['B'], 10)
    c.drawString(MARG, ALT * 0.62 + 3.4 * cm, 'RELATÓRIO DE ANÁLISE DE NEGÓCIO  ·  CIÊNCIA DE DADOS')
    c.setFillColor(BRANCO)
    c.setFont(F['K'], 34)
    c.drawString(MARG, ALT * 0.62 + 1.9 * cm, 'Detecção de Fraude')
    c.drawString(MARG, ALT * 0.62 + 0.55 * cm, 'em Pagamentos Online')
    c.setFont(F['L'], 14)
    c.setFillColor(colors.HexColor('#CFE0F3'))
    c.drawString(MARG, ALT * 0.62 - 0.6 * cm, 'Do score de risco ao lucro: modelo, regra de decisão e perfis de ataque')
    c.setFont(F['R'], 10.5)
    c.setFillColor(BRANCO)
    y = ALT * 0.36 - 1.4 * cm
    for linha in ['Base: IEEE-CIS Fraud Detection (Vesta Corporation) · 590.540 transações',
                  'Modelo: LightGBM calibrado + threshold dinâmico por valor da transação',
                  'Validação: Out-of-Time (20% mais recentes, nunca vistos no treino)']:
        c.drawString(MARG, y, linha)
        y -= 0.6 * cm
    c.setFont(F['B'], 11)
    c.drawString(MARG, 2.6 * cm, AUTOR)
    c.setFont(F['R'], 9.5)
    c.setFillColor(colors.HexColor('#CFE0F3'))
    c.drawString(MARG, 2.0 * cm, f'{CARGO}  ·  {date.today().strftime("%m/%Y")}')
    c.restoreState()


def pagina(c, doc):
    c.saveState()
    c.setFillColor(MARINHO)
    c.rect(0, ALT - 0.5 * cm, LARG, 0.5 * cm, stroke=0, fill=1)
    c.setFillColor(CIANO)
    c.rect(0, ALT - 0.5 * cm, 3.2 * cm, 0.5 * cm, stroke=0, fill=1)
    c.setFont(F['B'], 7.5)
    c.setFillColor(CINZA)
    c.drawString(MARG, ALT - 1.25 * cm, 'DETECÇÃO DE FRAUDE  ·  RELATÓRIO DE ANÁLISE DE NEGÓCIO')
    c.setStrokeColor(LINHA)
    c.setLineWidth(0.6)
    c.line(MARG, 1.55 * cm, LARG - MARG, 1.55 * cm)
    c.setFont(F['R'], 8)
    c.drawString(MARG, 1.05 * cm, f'IEEE-CIS Fraud Detection  ·  {AUTOR}')
    c.setFont(F['B'], 8.5)
    c.setFillColor(PRINCIPAL)
    c.drawRightString(LARG - MARG, 1.05 * cm, f'{doc.page}')
    c.restoreState()


# ------------------------------------------------------------------------------ conteúdo
def conteudo():
    s = []
    s.append(NextPageTemplate('interna'))
    s.append(PageBreak())

    # ======================================================== SUMÁRIO EXECUTIVO
    s += secao(None, 'Sumário Executivo',
               'Fraude é só 3,5% das transações, mas destrói mais valor do que toda a receita das compras legítimas. '
               'Este relatório mostra como um modelo de risco, combinado com uma regra de decisão financeira, '
               'transforma prejuízo em lucro e onde ainda há dinheiro escapando.')
    s.append(kpis([
        ('−US$ 297 mil', 'Resultado sem modelo<br/>no período futuro (OOT)', CINZA),
        ('+US$ 35,9 mil', 'Lucro líquido com o modelo<br/>no mesmo período', CIANO),
        ('68%', 'das fraudes bloqueadas<br/>(Recall no OOT)', PRINCIPAL),
        ('87,6%', 'dos clientes legítimos<br/>aprovados sem atrito', MARINHO),
    ]))
    s.append(p('Principais conclusões', 'h2'))
    s += bullets([
        '<b>O problema é financeiro, não estatístico.</b> Para cada US$ 1 de receita gerada por clientes legítimos, '
        'a operação perde cerca de US$ 2 em chargebacks. Aprovar tudo ou bloquear tudo leva a prejuízos de '
        'US$ 1,2 milhão no período analisado.',
        '<b>O corte padrão de 50% não serve.</b> A decisão usa a probabilidade calibrada e um corte que varia com o '
        'valor da compra: mais tolerante em micro-compras e mais rigoroso em tickets altos, onde o chargeback pesa mais.',
        '<b>O modelo generaliza para o futuro.</b> No período Out-of-Time, que o modelo nunca viu, o GINI ficou em 0,75 '
        '(queda moderada de 4,9%), a população de scores permaneceu estável (PSI = 0,003) e a operação seguiu lucrativa.',
        '<b>Existem quatro perfis de ataque distintos.</b> O <i>Cartão Clonado</i> é 45% das fraudes e 63% do '
        'prejuízo; a <i>Botnet</i> é volumosa, mas barata; o <i>Roubo de Contas</i> e o <i>Ataque Inteligente</i> '
        'são menores e de ticket alto.',
        '<b>O ponto cego está no Cartão Clonado.</b> O modelo barra 88% da Botnet, mas apenas 47% do Cartão '
        'Clonado no período futuro, e esse único grupo responde por 86% de todo o chargeback que escapa.',
    ])
    s.append(callout('Recomendação principal',
                     'Colocar o modelo em produção com a regra dinâmica e priorizar a próxima evolução no grupo '
                     '<b>Cartão Clonado</b>: features de perfil de gasto individual do cartão (desvio do ticket em '
                     'relação ao histórico do próprio cartão) e monitoramento semanal do Recall desse grupo.'))

    # ======================================================== 1. PROBLEMA
    s.append(PageBreak())
    s += secao(1, 'O Problema de Negócio',
               'Uma empresa de pagamentos ganha uma pequena margem em cada compra aprovada e perde o valor inteiro de '
               'cada fraude que deixa passar. O desafio é decidir, em milissegundos, quem aprovar.')
    s.append(p('Os dados vêm da competição <b>IEEE-CIS Fraud Detection</b> (Vesta Corporation): 590.540 transações de '
               'e-commerce em cerca de seis meses, com 434 colunas brutas de transação, identidade digital e '
               'dispositivo, muitas delas anonimizadas. Apenas <b>20.663 transações (3,5%) são fraudes</b>.'))
    s.append(p('Matriz de custo da operação', 'h2'))
    s.append(p('Toda a análise usa uma regra financeira explícita, em dólares reais, em vez de métricas abstratas:'))
    s.append(tabela(['Situação', 'Decisão', 'Impacto financeiro'], [
        ['<b>VN</b> — compra legítima', 'Aprovar', '<b>+2%</b> do valor (taxa sobre a venda)'],
        ['<b>FP</b> — legítima bloqueada', 'Bloquear', '<b>−2%</b> (receita perdida e atrito com o cliente)'],
        ['<b>FN</b> — fraude aprovada', 'Aprovar', '<b>−100%</b> do valor (chargeback)'],
        ['<b>VP</b> — fraude bloqueada', 'Bloquear', '0 (prejuízo evitado)'],
    ], larguras=[1.3, 0.8, 1.9], esq=(2,)))
    s.append(p('O dilema do risco inerente', 'h2'))
    s.append(p('Sem nenhum modelo, as duas estratégias extremas são ruins. Nos 80% iniciais da base (Treino + Validação), '
               '<b>aprovar tudo</b> geraria US$ 1,22 milhão de receita, mas US$ 2,47 milhões em chargebacks: '
               'um prejuízo de <b>US$ 1,25 milhão</b>. <b>Bloquear tudo</b> zera a fraude, mas sacrifica toda a '
               'receita (−US$ 1,22 milhão). O valor do modelo está em encontrar o meio-termo lucrativo.'))
    s.append(figura('02_03_dilema_financeiro_risco_inerente',
                    'Figura 1 — Resultado financeiro das estratégias sem modelo (Treino + Validação).'))
    s.append(callout('Por que 3,5% importa tanto',
                     'Uma fraude custa 100% do valor; uma venda legítima rende 2%. Por isso, cada fraude aprovada '
                     'apaga o lucro de cerca de <b>50 vendas legítimas</b> de mesmo valor. O ponto de equilíbrio '
                     'teórico é bloquear quando a probabilidade calibrada de fraude passa de <b>3,85%</b>.'))

    # ======================================================== 2. DADOS
    s.append(PageBreak())
    s += secao(2, 'O Que os Dados Revelam',
               'Antes de modelar, a análise exploratória respondeu: quando, onde e como a fraude acontece?')
    s.append(figura_grade([('01_02_painel_fraude_bivariado', 0), ('01_02_painel_fraude_bivariado', 2)],
                          'Figura 2 — Taxa de fraude por hora do dia e por linha de produto (a linha tracejada é a '
                          'média global de 3,5%).'))
    s += bullets([
        '<b>Horário:</b> a fraude se concentra entre 3h e 11h, com pico às <b>7h (≈10,6%)</b>, três vezes a média. '
        'É o padrão de ataques automatizados rodando de madrugada.',
        '<b>Produto:</b> o canal <b>C</b> tem <b>11,7%</b> de fraude, quase seis vezes o canal W (2,0%). '
        'Produtos digitais, de entrega imediata, atraem testes de cartão.',
        '<b>Valor:</b> fraudes e compras legítimas têm faixas de valor sobrepostas (média próxima de US$ 120); '
        'o valor sozinho não separa, mas combinado com outros sinais é um dos principais motores do modelo.',
        '<b>Dados ausentes são sinal:</b> 279 de 312 variáveis numéricas tinham forte assimetria, e a quantidade de '
        'campos vazios por transação virou uma feature. Contas automatizadas e cadastros rasos deixam rastros de ausência.',
    ])
    s.append(p('Da base bruta à tabela de modelagem', 'h2'))
    s.append(p('As 434 colunas brutas foram enriquecidas com features de negócio e depois filtradas por um funil '
               'estatístico: Information Value (remoção de variáveis sem poder preditivo), correlação entre pares '
               '(|r| &gt; 0,75, mantendo a de maior IV) e diagnóstico de multicolinearidade. O resultado foi uma '
               'tabela com <b>152 colunas</b> (136 numéricas e 13 categóricas).'))
    s.append(tabela(['Família de features', 'Exemplos', 'O que capturam'], [
        ['Velocidade (bloco C)', '<i>vel_C1 … vel_C14</i>', 'Quantos cartões, e-mails e endereços se ligam ao pagador'],
        ['Deltas de tempo (bloco D)', '<i>delta_D1, delta_D15</i>', 'Dias desde eventos anteriores: contas "recém-nascidas"'],
        ['Validação cadastral (bloco M)', '<i>match_cont_T/F</i>', 'Quantas checagens de dados o cliente passou ou falhou'],
        ['Identidade e dispositivo', '<i>disp_marca, seg_*, hist_*</i>', 'Marca, navegador, tela e histórico do aparelho'],
        ['Frequência (pós-split)', '<i>card1_freq, addr1_freq</i>', 'Raridade da faixa de cartão, endereço e e-mail'],
    ], larguras=[1.25, 1.2, 2.0], esq=(2,)))
    s.append(p('<b>Sem vazamento do futuro:</b> contagens de frequência calculadas na base inteira "enxergam" '
               'transações que ainda não existiam no momento da decisão; por isso foram descartadas e recalculadas '
               'apenas com o período de treino.'))
    s.append(p('O valor da transação: sobreposição, não separação', 'h2'))
    s.append(p('Em escala logarítmica, as compras legítimas se concentram em picos de preço típicos (perto de US$ 55 e '
               'US$ 110), enquanto as fraudes formam uma distribuição mais espalhada, com mais peso nas compras '
               'pequenas e na cauda de valores altos. As duas curvas se sobrepõem quase inteiramente: o valor '
               'sozinho não identifica uma fraude. Mas o <b>custo</b> de cada erro cresce com ele, e é por isso que a '
               'regra de decisão do Capítulo 4 passa a exigir mais rigor justamente nos tickets altos.'))
    s.append(figura('01_02_painel_fraude_bivariado',
                    'Figura 3 — Distribuição de log(1 + valor da transação) para fraudes e compras legítimas.',
                    largura=11.0 * cm, max_alt=9.0 * cm, painel=1))

    # ======================================================== 3. METODOLOGIA
    s.append(PageBreak())
    s += secao(3, 'Como o Modelo Foi Construído',
               'O desenho da validação imita a produção: treinar com o passado e medir no futuro.')
    s.append(tabela(['Janela', 'Período', 'Transações', 'Taxa de fraude', 'Uso'], [
        ['<b>Treino</b>', '60% iniciais', '354.324', '3,38%', 'Ajuste, tuning, calibração e regra de decisão'],
        ['<b>Validação</b>', '20% seguintes', '118.108', '3,90%', 'Comparação de modelos e políticas'],
        ['<b>Out-of-Time</b>', '20% finais', '118.108', '3,44%', 'Prova final, isolada até a avaliação'],
    ], larguras=[0.9, 0.9, 0.8, 0.8, 2.4], esq=(4,)))
    s += bullets([
        '<b>Validação cruzada por mês</b> (GroupKFold): cada mês fica inteiro em um único fold, evitando que '
        'transações do mesmo período apareçam no treino e no teste.',
        '<b>Seleção de algoritmo:</b> Regressão Logística, LightGBM, XGBoost e CatBoost foram comparados. O LightGBM '
        'teve a melhor discriminação na Validação (GINI 0,804) e o menor sobreajuste entre os modelos de árvore.',
        '<b>Poda por importância (SHAP):</b> o modelo foi retreinado mantendo apenas as N variáveis mais relevantes. '
        'Com <b>30 das 150 features</b>, o GINI caiu só 0,003, dentro da tolerância definida.',
        '<b>Calibração isotônica:</b> os scores do LightGBM ficavam inflados (média de 22% para uma taxa real de 3,4%). '
        'A calibração transforma o score em probabilidade real, requisito para comparar risco com custo.',
    ])
    s.append(figura('02_02_curva_poda_features_shap',
                    'Figura 4 — GINI na Validação conforme o número de features: 30 variáveis preservam quase todo '
                    'o poder do modelo completo.'))
    s.append(callout('Escolha consciente: modelo enxuto em vez do mais lucrativo na Validação',
                     'A versão com 125 features (<i>LGBM_Max</i>) lucrou cerca de US$ 6,3 mil a mais na Validação. '
                     'A versão com 30 features (<i>LGBM_Otimizado</i>) foi escolhida por <b>parcimônia</b>: um quarto '
                     'das variáveis, menos sobreajuste e menor exposição a mudanças nos dados em produção. O teste '
                     'Out-of-Time (Capítulo 6) confirma que ela se sustenta no futuro.', cor=PRINCIPAL))

    # ======================================================== 4. DECISÃO
    s.append(PageBreak())
    s += secao(4, 'Do Score à Decisão: a Regra que Gera Lucro',
               'Um bom modelo com a regra de corte errada ainda perde dinheiro. A decisão foi otimizada em dólares.')
    s.append(p('Três políticas foram comparadas na Validação, todas definidas apenas com dados de treino:'))
    s.append(tabela(['Política', 'Lucro líquido', 'Aprovação de legítimas', 'Fraudes barradas', 'Chargeback'], [
        ['Sem modelo (aprovar tudo)', '−US$ 494.022', '100,0%', '0,0%', 'US$ 796.120'],
        ['Corte clássico de 50%', 'US$ 167', '90,3%', '71,8%', 'US$ 232.749'],
        ['Corte fixo ótimo (3,41%)', 'US$ 24.631', '85,7%', '78,6%', 'US$ 177.633'],
        ['<b>Corte dinâmico por valor</b>', '<b>US$ 27.906</b>', '<b>88,4%</b>', '<b>75,0%</b>', '<b>US$ 176.930</b>'],
    ], larguras=[1.7, 1.0, 1.1, 1.0, 1.0], destaque=[3]))
    s.append(p('O <b>corte dinâmico</b> adapta a exigência ao valor da compra: começa em 5,1% de probabilidade para '
               'micro-compras e cai até 2,6% em tickets altos. Na prática, o cliente que compra pouco é incomodado '
               'menos, e as compras grandes, onde um chargeback custa mais, são vigiadas com mais rigor.'))
    s.append(figura('02_09_threshold_dinamico_lgbm_otimizado',
                    'Figura 5 — Probabilidade de fraude exigida para bloquear, conforme o valor da transação '
                    '(linha sólida) e o corte fixo de referência (tracejada).'))
    s.append(callout('Resultado da regra dinâmica',
                     'Frente ao corte fixo, a regra dinâmica lucrou <b>US$ 3,3 mil a mais</b> e aprovou <b>2,7 pontos '
                     'percentuais mais clientes legítimos</b> na Validação. Menos bloqueios injustos significam menos '
                     'abandono de carrinho, menos chamados no atendimento e menos revisão manual.'))

    # ======================================================== 5. ARQUÉTIPOS
    s.append(PageBreak())
    s += secao(5, 'Quem São os Fraudadores',
               'O modelo responde "qual a probabilidade". A segmentação responde "quem está atacando e como".')
    s.append(p('As 11.988 fraudes do período de treino foram agrupadas por comportamento (UMAP + K-Means, 12 clusters) '
               'e consolidadas em <b>quatro arquétipos de ataque</b>, cada um com uma assinatura operacional clara:'))
    s.append(tabela(['Arquétipo', '% das fraudes', '% do prejuízo', 'Ticket médio', 'Assinatura'], [
        ['<b>1. Cartão Clonado</b>', '44,8%', '<b>63,2%</b>', 'US$ 198', 'Varejo (W), débito, sem registro de dispositivo, baixa velocidade'],
        ['<b>2. Botnet</b>', '37,2%', '14,2%', 'US$ 53', 'Produto digital (C), velocidade extrema, tickets baixos'],
        ['<b>3. Roubo de Contas</b>', '13,4%', '17,8%', 'US$ 185', 'Apple + crédito, produtos H/R, se passa por cliente real'],
        ['<b>4. Ataque Inteligente</b>', '4,6%', '4,8%', 'US$ 148', 'Produto de alto risco (H) via Windows, em rajadas'],
    ], larguras=[1.25, 0.75, 0.75, 0.75, 2.5], esq=(4,)))
    s.append(figura('03_08_arquetipos_volume_vs_prejuizo',
                    'Figura 6 — Participação de cada arquétipo no volume de fraudes e no prejuízo em dólares.'))
    s += bullets([
        '<b>Cartão Clonado é o grande destruidor de valor:</b> com 45% das fraudes, concentra 63% do prejuízo '
        '(alavancagem de 1,4x). Opera todos os dias e <b>cresce 8% ao mês</b> no fim do período de treino.',
        '<b>Botnet é volume, não valor:</b> quase tantos ataques quanto o Cartão Clonado, mas só 14% do prejuízo. '
        'Sua assinatura de velocidade é fácil de reconhecer.',
        '<b>Roubo de Contas pesa mais do que aparenta:</b> 13% das fraudes e 18% do prejuízo, com tickets altos e '
        'dispositivos premium, imitando clientes legítimos.',
        '<b>Ataque Inteligente age em ondas:</b> ativo em 85% dos dias, com picos de até 7x a mediana diária, e em '
        'forte queda (−29% ao mês), sinal de que migrou de tática ou foi contido.',
    ])
    s.append(figura('03_10_arquetipos_evolucao_temporal',
                    'Figura 7 — Volume diário por arquétipo e participação no prejuízo semanal: o Cartão Clonado '
                    'responde por 47% a 78% das perdas.', max_alt=13.0 * cm))
    s.append(p('Resposta operacional por arquétipo', 'h2'))
    s.append(p('Cada perfil pede uma defesa diferente. A segmentação transforma o score em ações concretas para as '
               'áreas de risco, produto e atendimento:'))
    s.append(tabela(['Arquétipo', 'Como reconhecer', 'Resposta recomendada'], [
        ['<b>1. Cartão Clonado</b>', 'Compra de varejo de valor alto, sem dados de dispositivo, cartão de débito',
         'Exigir autenticação adicional (3-D Secure) acima de um valor de referência e revisar o desvio do ticket '
         'frente ao histórico do cartão'],
        ['<b>2. Botnet</b>', 'Muitas transações pequenas em sequência no produto digital C, alta velocidade',
         'Limites de velocidade por cartão, e-mail e endereço; bloqueio automático já é eficaz (Recall de 88%)'],
        ['<b>3. Roubo de Contas</b>', 'Dispositivo premium e crédito, mas conta ou cartão com histórico curto',
         'Confirmar mudanças recentes de cadastro e dispositivo antes de compras de valor alto'],
        ['<b>4. Ataque Inteligente</b>', 'Produto H via Windows, chega em ondas concentradas em poucos dias',
         'Alerta de pico diário por produto para reagir rápido às rajadas'],
    ], larguras=[1.2, 1.9, 2.4], esq=(1, 2)))

    # ======================================================== 6. OOT
    s.append(PageBreak())
    s += secao(6, 'O Teste do Futuro',
               'A pergunta decisiva: o modelo continua discriminando, decidindo e gerando lucro em meses que nunca viu?')
    s.append(tabela(['Indicador', 'Validação', 'In-Time (Treino + Validação)', 'Out-of-Time', 'Variação'], [
        ['GINI', '0,809', '0,786', '<b>0,747</b>', '−4,9%'],
        ['KS', '0,648', '0,625', '<b>0,577</b>', '−7,6%'],
        ['Fraudes barradas (Recall)', '75,0%', '75,1%', '<b>68,0%</b>', '−7,1 p.p.'],
        ['Aprovação de legítimas', '88,4%', '86,9%', '<b>87,6%</b>', '+0,8 p.p.'],
        ['Lucro líquido', 'US$ 27.906', 'US$ 262.453', '<b>US$ 35.861</b>', '—'],
        ['Lucro por transação', 'US$ 0,236', 'US$ 0,556', '<b>US$ 0,304</b>', '+29% vs. Validação'],
    ], larguras=[1.5, 0.9, 1.4, 1.0, 1.1], destaque=[4]))
    s.append(figura('04_02_roc_pr_in_time_vs_oot',
                    'Figura 8 — Curvas ROC e Precision-Recall: a curva do período futuro acompanha de perto a '
                    'do período conhecido.'))
    s += bullets([
        '<b>Ordenação preservada:</b> os 10% de transações com maior score concentram <b>61% de todas as fraudes</b> '
        '(6,1 vezes a taxa média), sem nenhuma inversão na escada de risco.',
        '<b>Valor gerado:</b> no período futuro, a operação sem modelo perderia US$ 297 mil. Com o modelo, '
        'lucrou US$ 35,9 mil, uma diferença de <b>US$ 333 mil</b> em cerca de seis semanas, evitando '
        'US$ 441 mil em fraudes (72% do valor fraudado).',
        '<b>Experiência do cliente estável:</b> 87,6% dos clientes legítimos aprovados, e a taxa de bloqueio (14,3%) '
        'fica entre a da Validação e a do Treino, sem sobrecarregar o time de revisão.',
    ])
    s.append(callout('Leitura honesta da queda',
                     'O Recall caiu 7 pontos no período futuro. A queda não vem de uma mudança na taxa de fraude '
                     '(estável em 3,4–3,5%), nem da população de scores (PSI = 0,003), e sim de uma perda real de '
                     'discriminação concentrada em um único grupo, analisado a seguir.', cor=PRINCIPAL))

    # ======================================================== 7. ONDE FALHA
    s.append(PageBreak())
    s += secao(7, 'Onde o Modelo Falha',
               'Cruzar o desempenho com os arquétipos mostra exatamente para onde o dinheiro escapa.')
    s.append(tabela(['Arquétipo', 'Recall In-Time', 'Recall OOT', 'Prejuízo vazado (OOT)', '% do vazamento'], [
        ['<b>1. Cartão Clonado</b>', '60,5%', '<b>46,7%</b>', '<b>US$ 145.589</b>', '<b>86,3%</b>'],
        ['2. Botnet', '89,3%', '87,7%', 'US$ 8.813', '5,2%'],
        ['3. Roubo de Contas', '80,9%', '77,3%', 'US$ 10.863', '6,4%'],
        ['4. Ataque Inteligente', '81,6%', '80,6%', 'US$ 3.396', '2,0%'],
    ], larguras=[1.4, 0.9, 0.9, 1.2, 0.9], destaque=[0]))
    s.append(figura('04_13_recall_vazamento_grupos',
                    'Figura 9 — Recall por arquétipo, In-Time versus Out-of-Time: a queda se concentra no Cartão '
                    'Clonado (o prejuízo vazado em US$ está na tabela acima).', largura=13.5 * cm, painel=0))
    s += bullets([
        '<b>Forte contra automação:</b> Botnet, Roubo de Contas e Ataque Inteligente mantêm Recall entre 77% e 88% no '
        'futuro, e juntos respondem por apenas 14% do vazamento.',
        '<b>Fraco no Cartão Clonado:</b> o score mediano dessas fraudes caiu de 0,54 para 0,42 e passou a ficar '
        'dentro da faixa de corte; mais da metade delas é aprovada. Sem registro de dispositivo e sem picos de '
        'velocidade, essas transações se parecem com compras legítimas de alto valor.',
        '<b>O problema está no score, não na regra:</b> o corte já é o mais rigoroso justamente nos tickets altos. '
        'A solução passa por novas informações sobre o comportamento individual do cartão.',
    ])
    s.append(callout('Implicação para o negócio',
                     'O grupo que mais destrói valor é também o que estava crescendo no fim do treino. Cada ponto de '
                     'Recall recuperado no Cartão Clonado vale mais do que melhorias equivalentes em qualquer outro grupo.',
                     cor=CIANO))

    # ======================================================== 8. ESTABILIDADE
    s.append(PageBreak())
    s += secao(8, 'Estabilidade e Monitoramento',
               'Modelos de fraude envelhecem. O que observar para saber quando agir?')
    s += bullets([
        '<b>Taxa de fraude:</b> 3,51% (In-Time) e 3,44% (OOT). A diferença não é estatisticamente significativa '
        '(p = 0,23).',
        '<b>Distribuição do score:</b> PSI de 0,003 entre In-Time e OOT, bem abaixo do limite de alerta (0,10).',
        '<b>Features do modelo:</b> 29 das 30 variáveis estáveis; apenas <i>match_cont_F</i> (falhas cadastrais) em '
        'alerta (PSI 0,127), e ela tem baixa importância no modelo.',
        '<b>Motivos da decisão:</b> o ranking de importância SHAP entre In-Time e OOT tem correlação de 0,93; o modelo '
        'decide pelos mesmos motivos nas duas janelas.',
        '<b>Desempenho semanal:</b> o GINI de todas as semanas do OOT ficou acima do piso de aceitação (0,709), com '
        'tendência levemente positiva, e todas as semanas foram lucrativas.',
    ])
    s.append(figura('04_12_gini_temporal_semanal',
                    'Figura 10 — GINI semana a semana ao longo de Treino, Validação e Out-of-Time, com o piso de '
                    'aceitação.'))
    s.append(p('Aplicação na base cega do Kaggle', 'h2'))
    s.append(p('O mesmo artefato escorou as <b>506.691 transações</b> da base de teste da competição, sem rótulo. '
               'Todas as 30 features ficaram estáveis frente ao período de treino, o PSI do score foi de 0,006 e a taxa '
               'de bloqueio de 12,6%, próxima da Validação e do OOT (≈14%), sinal de operação consistente.'))
    s.append(callout('Lição de engenharia',
                     'A auditoria de estabilidade da base cega detectou uma divergência de paridade: e-mails ausentes eram '
                     'representados de forma diferente no treino e na inferência, e uma feature disparava como "crítica". '
                     'Corrigido o pipeline, todas as features voltaram à faixa estável. Monitorar o PSI da inferência '
                     'protege o negócio também contra erros de engenharia, não só contra mudanças no mundo.',
                     cor=PRINCIPAL))

    # ======================================================== 9. HIPÓTESE
    s.append(PageBreak())
    s += secao(9, 'Hipótese Testada e Descartada',
               'Um corte diferente para cada grupo de clientes aumentaria o lucro?')
    s.append(p('A hipótese era intuitiva: agrupar todas as transações (legítimas e fraudes) em segmentos e otimizar um '
               'corte específico para cada um. Foram testadas 6 variações, com 4 e 10 grupos, cortes fixos e curvas '
               'dinâmicas por grupo, sempre ajustadas no treino e avaliadas na Validação.'))
    s.append(tabela(['Estratégia', 'Parâmetros', 'Lucro no treino', 'Lucro na Validação', 'vs. regra atual'], [
        ['<b>Dinâmico global (atual)</b>', '3', 'US$ 234.548', '<b>US$ 27.906</b>', '—'],
        ['Fixo global', '1', 'US$ 231.752', 'US$ 24.631', '−11,7%'],
        ['Fixo por grupo (4 grupos)', '4', 'US$ 246.882', 'US$ 19.166', '−31,3%'],
        ['Fixo por grupo (10 grupos)', '10', 'US$ 256.954', 'US$ 16.987', '−39,1%'],
        ['Dinâmico por grupo (4 grupos)', '12', 'US$ 250.512', 'US$ 14.405', '−48,4%'],
        ['Dinâmico por grupo (10 grupos)', '30', 'US$ 262.218', 'US$ 12.980', '−53,5%'],
    ], larguras=[1.8, 0.8, 1.0, 1.1, 0.9], destaque=[0]))
    s.append(p('O padrão é claro: <b>quanto mais parâmetros ajustados, maior o lucro aparente no treino e menor o lucro '
               'real em dados novos</b>. Os cortes por grupo memorizam ruído; o score calibrado já incorpora o que '
               'diferencia os segmentos. A hipótese foi descartada e a regra global mantida.'))
    s.append(figura('03_13_benchmark_grupamento_vs_exponencial',
                    'Figura 11 — Lucro líquido de cada estratégia na Validação: todas as variações por grupo '
                    'ficam abaixo da regra global atual.', largura=14.5 * cm, painel=0))

    # ======================================================== 10. RECOMENDAÇÕES
    s.append(PageBreak())
    s += secao(10, 'Recomendações e Próximos Passos',
               'O modelo está pronto para operar. As ações abaixo estão ordenadas por impacto esperado.')
    s.append(tabela(['Prioridade', 'Ação', 'Por quê', 'Impacto esperado'], [
        ['<b>1</b>', 'Features de perfil de gasto do cartão (desvio do ticket vs. histórico do próprio cartão)',
         'Cartão Clonado gera 86% do vazamento e não tem sinais de velocidade ou dispositivo', 'Alto'],
        ['<b>2</b>', 'Monitorar semanalmente o Recall por arquétipo e o PSI das features',
         'Detectar cedo a degradação no grupo que mais cresce', 'Alto'],
        ['<b>3</b>', 'Recalibração periódica do score com a safra mais recente',
         'O modelo subestima a fraude fora do treino (−15% no OOT)', 'Médio'],
        ['<b>4</b>', 'Testar o modelo de 125 features no OOT',
         'Confirmar se a parcimônia compensa o lucro extra visto na Validação', 'Médio'],
        ['<b>5</b>', 'Fila de revisão manual priorizada por valor para a faixa de score do Cartão Clonado',
         'Recuperar parte do prejuízo sem bloquear automaticamente bons clientes', 'Médio'],
    ], larguras=[0.6, 2.1, 2.2, 0.8], esq=(1, 2)))
    s.append(p('Limitações', 'h2'))
    s += bullets([
        'As variáveis são parcialmente anonimizadas pela Vesta; parte dos sinais fortes (ex.: <i>V70</i>) não tem '
        'interpretação de negócio direta.',
        'A regra de custo (+2% / −100%) é uma simplificação: custos operacionais de revisão e de atrito não foram '
        'modelados explicitamente.',
        'Os arquétipos foram definidos com fraudes do treino; padrões realmente novos no futuro são atribuídos ao '
        'grupo mais parecido.',
        'O Information Value foi calculado na base inteira antes do split; o corte usado é baixo, mas o ideal é '
        'calculá-lo só no treino.',
    ])
    s.append(callout('Reprodutibilidade',
                     'Todo o pipeline pode ser reproduzido sem notebooks: <b>python main.py</b> '
                     'executa preparação, treino, segmentação, avaliação e escoragem a partir do '
                     '<b>configs.yaml</b>, com versões fixadas e experimentos isolados. '
                     'Os detalhes estão no README do repositório.', cor=PRINCIPAL))

    # ======================================================== GLOSSÁRIO
    s.append(CondPageBreak(6.5 * cm))
    s.append(p('Apêndice · Glossário', 'h2'))
    s.append(tabela(['Termo', 'Significado'], [
        ['GINI / ROC-AUC', 'Capacidade do score de ordenar fraudes acima de legítimas (GINI = 2 × AUC − 1)'],
        ['KS', 'Maior distância entre as distribuições de score de fraudes e de legítimas'],
        ['Recall', 'Percentual das fraudes que o modelo bloqueia'],
        ['Precision', 'Percentual dos bloqueios que eram de fato fraude'],
        ['PSI', 'Índice de estabilidade populacional: &lt; 0,10 estável; 0,10–0,25 alerta; ≥ 0,25 crítico'],
        ['Out-of-Time (OOT)', 'Período mais recente, separado e nunca usado no desenvolvimento do modelo'],
        ['Calibração', 'Ajuste que transforma o score do modelo em probabilidade real de fraude'],
        ['SHAP', 'Método que mede quanto cada variável contribui para a decisão do modelo'],
    ], larguras=[1.0, 3.5], esq=(1,)))
    s.append(Spacer(1, 14))
    s.append(cartao_autor())
    return s


def main() -> Path:
    doc = BaseDocTemplate(
        str(SAIDA), pagesize=A4, leftMargin=MARG, rightMargin=MARG, topMargin=2.0 * cm, bottomMargin=2.1 * cm,
        title='Detecção de Fraude em Pagamentos Online — Relatório de Análise de Negócio',
        author=AUTOR, subject='IEEE-CIS Fraud Detection',
    )
    frame = Frame(MARG, 2.1 * cm, UTIL, ALT - 4.1 * cm, id='f', leftPadding=0, rightPadding=0, topPadding=0,
                  bottomPadding=0)
    doc.addPageTemplates([
        PageTemplate(id='capa', frames=[frame], onPage=capa),
        PageTemplate(id='interna', frames=[frame], onPage=pagina),
    ])
    try:
        doc.build(conteudo())
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)
    return SAIDA


if __name__ == '__main__':
    print(f'Relatório gerado: {main()}')
