#!/usr/bin/env python3
"""Passo 4a (obrigatório em música): confere o início de cada frase pelo que é ouvido.

Uso:
    python conferir_letras.py trabalho/voz.wav trabalho/frases_sinc.json

Lê a voz isolada letra por letra com o modelo de português, SEM usar a letra da
música (decodificação livre), e mostra, ao lado do início de cada frase, os
grupos de letras ouvidos por perto e quando começam. Como não é forçado a seguir
o texto, não "espreme" palavras em notas longas — foi assim que achamos uma
ponte inteira deslocada 5 s ("todo" ouvido como "trodo" aos 74,2 s, quando a
legenda o punha aos 68,7 s).

Leia a lista inteira: a 1ª palavra da frase deve aparecer (mesmo mal escrita:
"trodo" = todo, "gratu" = grato, "céu" = seu) perto do início marcado. Se ela
só aparece bem depois, corrija o "inicio" no frases_sinc.json para o tempo
ouvido menos 0,2 s (a antecipação) e ajuste o "fim" da frase anterior.
"""
import argparse
import json
import sys

from audio import ler_audio
from sincronizar import ANTECIPA, QUADRO, carregar_modelo, emissoes


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("voz")
    p.add_argument("frases")
    p.add_argument("--janela", type=float, default=1.2, help="segundos antes/depois do início a mostrar")
    a = p.parse_args()

    sessao, vocab = carregar_modelo()
    inv = {v: k for k, v in vocab.items()}
    melhor = emissoes(sessao, ler_audio(a.voz)).argmax(1)
    grupos, g, gt, ant = [], [], None, -1
    for i, x in enumerate(melhor):
        if x != ant and x != 0:
            c = inv[int(x)]
            if c == "|":
                if g:
                    grupos.append((gt, "".join(g)))
                    g = []
            else:
                if not g:
                    gt = i * QUADRO
                g.append(c)
        ant = x
    if g:
        grupos.append((gt, "".join(g)))

    frases = json.load(open(a.frases, encoding="utf-8"))["frases"]
    for k, f in enumerate(frases, 1):
        s = f["inicio"] + ANTECIPA
        perto = [f"{t:.2f}:{w}" for t, w in grupos if s - a.janela <= t <= s + a.janela]
        print(f"{k:3d} {s:7.2f}  {f['texto'][:24]:24s} | ouvido: {'  '.join(perto) or '(nada)'}")


if __name__ == "__main__":
    main()
