"""Funções compartilhadas: carregar o estilo, achar a fonte e medir o texto.

Todos os scripts usam este módulo para que a quebra das frases (segmentar.py)
e o desenho no vídeo (legendar.py) concordem sobre o tamanho de cada frase.
"""
import json
import os
from pathlib import Path

from fontTools.ttLib import TTCollection, TTFont
from PIL import ImageFont

SKILL_DIR = Path(__file__).resolve().parent.parent
ESTILO_PADRAO = SKILL_DIR / "assets" / "estilo.json"
LARGURA_BASE = 1080  # o design do Figma foi feito em 1080x1920


def carregar_estilo(caminho=None):
    caminho = Path(caminho) if caminho else ESTILO_PADRAO
    with open(caminho, encoding="utf-8") as f:
        estilo = json.load(f)
    fonte = Path(os.path.expanduser(estilo["fonte_arquivo"]))
    if not fonte.is_absolute():
        fonte = (caminho.parent / fonte).resolve()
    estilo["fonte_arquivo"] = str(fonte)
    estilo.setdefault("fonte_indice", 0)
    return estilo


def aplicar_fonte(estilo, arquivo, nome=None):
    """Troca a fonte do estilo (ex.: a Helvetica Neue do usuário).

    `arquivo` pode ser .ttf, .otf ou .ttc. Em .ttc (coleção, comum no Mac),
    `nome` escolhe a face, ex.: "Condensed Bold".
    """
    arquivo = str(Path(os.path.expanduser(arquivo)).resolve())
    indice = 0
    if arquivo.lower().endswith(".ttc"):
        faces = TTCollection(arquivo).fonts
        nomes = [_nome_completo(f) for f in faces]
        if nome:
            alvo = nome.lower().replace("-", " ")
            achados = [i for i, n in enumerate(nomes) if alvo in n.lower().replace("-", " ")]
            if not achados:
                raise SystemExit(f"Face '{nome}' não encontrada em {arquivo}. Faces: {nomes}")
            indice = achados[0]
    estilo["fonte_arquivo"] = arquivo
    estilo["fonte_indice"] = indice
    estilo["escala_horizontal"] = 100  # a compensação de largura só vale para a fonte reserva
    return estilo


def _nome_completo(font):
    return font["name"].getDebugName(4) or font["name"].getDebugName(1)


def info_fonte(estilo):
    """Nome que o libass reconhece e fator de tamanho CSS -> ASS."""
    arquivo, indice = estilo["fonte_arquivo"], estilo["fonte_indice"]
    font = TTFont(arquivo, fontNumber=indice) if arquivo.lower().endswith(".ttc") else TTFont(arquivo)
    nome = _nome_completo(font)
    os2, upem = font["OS/2"], font["head"].unitsPerEm
    # O libass dimensiona a fonte para que (winAscent + winDescent) = tamanho ASS;
    # no Figma/CSS o tamanho é o "em". Este fator converte um no outro.
    fator = (os2.usWinAscent + os2.usWinDescent) / upem
    return {"nome": nome, "fator_ass": fator, "pasta": str(Path(arquivo).parent)}


def escala(estilo, largura_video):
    return largura_video / LARGURA_BASE


def preparar_texto(estilo, texto):
    texto = " ".join(texto.split())
    return texto.upper() if estilo.get("caixa_alta", True) else texto


class Medidor:
    """Mede a largura de uma frase já com caixa alta e espaçamento entre letras."""

    def __init__(self, estilo, largura_video=LARGURA_BASE):
        self.estilo = estilo
        self.k = escala(estilo, largura_video)
        tamanho = estilo["tamanho_px"] * self.k
        self.fonte = ImageFont.truetype(estilo["fonte_arquivo"], size=max(1, round(tamanho * 4)),
                                        index=estilo["fonte_indice"])
        self.fator_pil = tamanho / round(tamanho * 4)
        self.espaco = estilo.get("espacamento_letras_px", 0) * self.k
        self.limite = estilo.get("largura_maxima", 0.84) * largura_video
        self.sx = estilo.get("escala_horizontal", 100) / 100

    def largura(self, texto):
        texto = preparar_texto(self.estilo, texto)
        if not texto:
            return 0.0
        base = self.fonte.getlength(texto) * self.fator_pil
        return base * self.sx + self.espaco * (len(texto) - 1)

    def cabe(self, texto):
        return self.largura(texto) <= self.limite
