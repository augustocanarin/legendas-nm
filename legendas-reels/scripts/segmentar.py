#!/usr/bin/env python3
"""Passo 2: agrupa as palavras transcritas em frases curtas, uma linha cada.

Uso:
    python segmentar.py transcricao.json -o frases.json [--largura-video 1080]

Cada frase precisa caber numa única linha centralizada, no tamanho do estilo.
A quebra respeita: fim de frase (. ? !), pausas na fala, vírgulas quando a
frase já está longa, e nunca deixa "de", "que", "o"... pendurado no fim.

O frases.json é para ser revisado: corrija o "texto" à vontade; os tempos
de cada palavra ficam em "palavras" caso precise dividir uma frase.
"""
import argparse
import json
import re
import sys

from estilo import Medidor, carregar_estilo

PAUSA_QUEBRA = 0.5     # silêncio (s) que força uma nova frase
EMENDA = 0.35          # buraco menor que isso entre frases é emendado (evita piscar)
SOBRA = 0.25           # quanto a frase fica na tela depois da última palavra
DURACAO_MINIMA = 0.6

PALAVRAS_FRACAS = set("""
a o as os um uma uns umas e ou mas nem que se de do da dos das no na nos nas em
ao aos à às pelo pela pelos pelas por para pra pro com sem sob sobre entre até
meu minha teu tua seu sua nosso nossa lhe me te nos vos é não tão mais muito
quando como onde porque pois já
""".split())

FIM_FRASE = re.compile(r"[.?!…]+[\"”')]*$")
PAUSA_TEXTO = re.compile(r"[,;:—–-][\"”')]*$")


def limpa(p):
    return re.sub(r"[^\wÀ-ÿ]", "", p.lower())


def texto_de(palavras):
    return " ".join(w["palavra"] for w in palavras)


def blocos(palavras, usar_pausas=True):
    """Corta onde a quebra é obrigatória: fim de frase (. ? !) ou pausa na fala."""
    atual = []
    for w in palavras:
        if usar_pausas and atual and w["inicio"] - atual[-1]["fim"] > PAUSA_QUEBRA:
            yield atual
            atual = []
        atual.append(w)
        if FIM_FRASE.search(w["palavra"]) or w.get("quebra"):   # "quebra" = fim de verso da letra
            yield atual
            atual = []
    if atual:
        yield atual


def custo_linha(linha, ultima, sozinha, medidor):
    largura = medidor.largura(texto_de(linha))
    if largura > medidor.limite and len(linha) > 1:
        return float("inf")
    custo = 10.0                                   # cada frase a mais na tela custa: prefere menos cortes
    custo += 4 * (1 - min(1.0, largura / medidor.limite)) ** 2   # frases equilibradas
    fim = linha[-1]["palavra"]
    if not ultima:
        if PAUSA_TEXTO.search(fim):
            custo -= 4                             # vírgula é um ótimo lugar para cortar
        elif limpa(fim) in PALAVRAS_FRACAS:
            custo += 8                             # evita "...com a" / "de" pendurado no fim
        if limpa(linha[0]["palavra"]) in PALAVRAS_FRACAS and len(linha) == 1:
            custo += 4
    # vírgula no meio da frase = um corte natural foi desperdiçado
    custo += 9 * sum(1 for w in linha[:-1] if PAUSA_TEXTO.search(w["palavra"]))
    if len(linha) == 1 and not sozinha:
        custo += 7                                 # palavra sozinha na tela só quando faz sentido
    return custo


def quebrar_bloco(bloco, medidor):
    """Escolhe as quebras de menor custo (programação dinâmica sobre as palavras)."""
    n = len(bloco)
    melhor = [0.0] + [float("inf")] * n
    volta = [0] * (n + 1)
    for j in range(1, n + 1):
        for i in range(j - 1, -1, -1):
            linha = bloco[i:j]
            if len(linha) > 1 and medidor.largura(texto_de(linha)) > medidor.limite:
                break
            c = melhor[i] + custo_linha(linha, j == n, n == 1, medidor)
            if c < melhor[j]:
                melhor[j], volta[j] = c, i
    linhas, j = [], n
    while j > 0:
        linhas.insert(0, bloco[volta[j]:j])
        j = volta[j]
    return linhas


def segmentar(palavras, medidor, usar_pausas=True):
    frases = []
    for bloco in blocos(palavras, usar_pausas):
        frases.extend(quebrar_bloco(bloco, medidor))
    return frases


def cronometrar(grupos):
    frases = [{"inicio": g[0]["inicio"], "fim": g[-1]["fim"], "texto": texto_de(g),
               "palavras": [[w["palavra"], w["inicio"], w["fim"]] for w in g]} for g in grupos]
    for i, f in enumerate(frases):
        proxima = frases[i + 1]["inicio"] if i + 1 < len(frases) else None
        fim = f["fim"] + SOBRA
        fim = max(fim, f["inicio"] + DURACAO_MINIMA)
        if proxima is not None:
            if proxima - fim < EMENDA:
                fim = proxima
            fim = min(fim, proxima)
        f["fim"] = round(fim, 3)
        f["inicio"] = round(f["inicio"], 3)
    return frases


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("transcricao")
    p.add_argument("-o", "--saida", default="frases.json")
    p.add_argument("--estilo", default=None)
    p.add_argument("--largura-video", type=int, default=1080)
    a = p.parse_args()

    with open(a.transcricao, encoding="utf-8") as f:
        dados = json.load(f)
    medidor = Medidor(carregar_estilo(a.estilo), a.largura_video)
    if "frases" in dados:
        # redividir frases já revisadas (ex.: depois de mudar o tamanho máximo da linha):
        # usa os tempos de cada palavra e mantém o fim de cada frase antiga como quebra
        palavras = []
        for f in dados["frases"]:
            ws = f.get("palavras") or [[p, f["inicio"], f["fim"]] for p in f["texto"].split()]
            for p, t0, t1 in ws:
                palavras.append({"palavra": p, "inicio": t0, "fim": t1})
            palavras[-1]["quebra"] = True
    else:
        palavras = dados["palavras"]
    # Em frases revisadas, os limites já são os certos; pausas de notas longas não devem cortar.
    frases = cronometrar(segmentar(palavras, medidor, usar_pausas="frases" not in dados))

    with open(a.saida, "w", encoding="utf-8") as f:
        json.dump({"frases": frases}, f, ensure_ascii=False, indent=1)
    for i, fr in enumerate(frases, 1):
        print(f"{i:3d}  {fr['inicio']:6.2f}-{fr['fim']:6.2f}  {fr['texto']}", file=sys.stderr)
    print(f"\n{len(frases)} frases -> {a.saida}", file=sys.stderr)


if __name__ == "__main__":
    main()
