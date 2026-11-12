#!/usr/bin/env python3
"""Passo 2b (músicas): troca o texto transcrito pela letra oficial, mantendo os tempos.

Uso:
    python alinhar_letra.py transcricao.json letra.txt -o transcricao_letra.json

A letra pode vir com versos repetidos uma vez só (refrão): cada trecho cantado é
comparado com todos os versos e recebe o que mais se parece, quantas vezes for
cantado. Cada verso da letra vira uma quebra de frase obrigatória na legenda.

Trechos que não batem com nenhum verso (fala, "ô", improviso) ficam com o texto
transcrito e aparecem na lista de "não encaixados" para revisão.
"""
import argparse
import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher

LIMIAR = 0.55   # semelhança mínima para aceitar um verso


def norm(p):
    p = unicodedata.normalize("NFD", p.lower())
    p = "".join(c for c in p if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]", "", p)


def ler_letra(caminho):
    versos = []
    for linha in open(caminho, encoding="utf-8"):
        linha = " ".join(linha.split())
        if linha and not linha.startswith("["):     # ignora marcações tipo [Refrão]
            palavras = linha.split()
            versos.append({"texto": linha, "palavras": palavras, "norm": [norm(p) for p in palavras]})
    return versos


def semelhanca(a, b):
    """Semelhança entre sequências de palavras, tolerando palavras mal ouvidas."""
    sm = SequenceMatcher(None, a, b, autojunk=False)
    iguais = sum(m.size for m in sm.get_matching_blocks())
    # palavras parecidas (ex.: "passa" x "basta") contam meio ponto
    parecidas = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            for x, y in zip(a[i1:i2], b[j1:j2]):
                if SequenceMatcher(None, x, y).ratio() >= 0.6:
                    parecidas += 0.5
    return 2 * (iguais + parecidas) / (len(a) + len(b)) if a and b else 0.0


def encaixar(palavras_t, verso, ini):
    """Acha a janela da transcrição que começa em `ini` e melhor corresponde ao verso."""
    alvo = verso["norm"]
    melhor = (0.0, ini)
    for tam in range(max(1, len(alvo) - 3), len(alvo) + 4):
        fim = ini + tam
        if fim > len(palavras_t):
            break
        s = semelhanca([norm(w["palavra"]) for w in palavras_t[ini:fim]], alvo)
        if s > melhor[0]:
            melhor = (s, fim)
    return melhor


def tempos_do_verso(verso, janela):
    """Dá a cada palavra da letra o tempo da palavra transcrita correspondente."""
    origem = [norm(w["palavra"]) for w in janela]
    destino = verso["norm"]
    tempos = [None] * len(destino)
    sm = SequenceMatcher(None, origem, destino, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
            for k in range(j2 - j1):
                w = janela[i1 + k]
                tempos[j1 + k] = (w["inicio"], w["fim"])
        elif tag == "replace":   # quantidades diferentes: espalha o trecho pelas palavras
            t0, t1 = janela[i1]["inicio"], janela[i2 - 1]["fim"]
            passo = (t1 - t0) / (j2 - j1)
            for k in range(j2 - j1):
                tempos[j1 + k] = (t0 + k * passo, t0 + (k + 1) * passo)
    # palavras da letra sem par na transcrição: interpola entre as vizinhas
    t_ini, t_fim = janela[0]["inicio"], janela[-1]["fim"]
    for k, t in enumerate(tempos):
        if t is None:
            antes = next((tempos[i][1] for i in range(k - 1, -1, -1) if tempos[i]), t_ini)
            depois = next((tempos[i][0] for i in range(k + 1, len(tempos)) if tempos[i]), t_fim)
            tempos[k] = (antes, max(antes, depois))
    return tempos


def alinhar(palavras_t, versos):
    saida, nao_encaixados, usados = [], [], []
    i = 0
    while i < len(palavras_t):
        candidatos = [(encaixar(palavras_t, v, i), n) for n, v in enumerate(versos)]
        (score, fim), n = max(candidatos, key=lambda c: c[0][0])
        if score >= LIMIAR:
            verso = versos[n]
            for palavra, (t0, t1) in zip(verso["palavras"], tempos_do_verso(verso, palavras_t[i:fim])):
                saida.append({"palavra": palavra, "inicio": round(t0, 3), "fim": round(t1, 3), "confianca": 1.0})
            saida[-1]["quebra"] = True
            usados.append((palavras_t[i]["inicio"], verso["texto"], round(score, 2)))
            i = fim
        else:
            w = dict(palavras_t[i])
            saida.append(w)
            nao_encaixados.append(w)
            i += 1
    return saida, usados, nao_encaixados


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("transcricao")
    p.add_argument("letra")
    p.add_argument("-o", "--saida", default="transcricao_letra.json")
    a = p.parse_args()

    dados = json.load(open(a.transcricao, encoding="utf-8"))
    versos = ler_letra(a.letra)
    palavras, usados, soltos = alinhar(dados["palavras"], versos)
    dados["palavras"] = palavras
    dados["letra_aplicada"] = True
    json.dump(dados, open(a.saida, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    for t, texto, s in usados:
        print(f"  {t:7.2f}s  ({s:.2f})  {texto}", file=sys.stderr)
    if soltos:
        print("\nNão encaixados (revise; podem ser fala, 'ô', improviso ou erro):", file=sys.stderr)
        for w in soltos:
            print(f"  {w['inicio']:7.2f}s  {w['palavra']}", file=sys.stderr)
    print(f"\n{len(usados)} versos encaixados -> {a.saida}", file=sys.stderr)


if __name__ == "__main__":
    main()
