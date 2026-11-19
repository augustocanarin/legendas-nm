#!/usr/bin/env python3
"""Passo 3c (recomendado em música): confere o início de cada frase por votação.

Uso:
    python refinar_tempos.py trabalho/voz.wav trabalho/frases_sinc.json -o trabalho/frases_final.json

Nenhum método sozinho acerta o tempo do canto: o alinhador (sincronizar.py) se
perde em notas longas e o Whisper "cola" o início de cada trecho no fim do
anterior. Este passo transcreve a voz em janelas curtas e sobrepostas (14 s,
andando 3,5 s), então cada palavra recebe várias estimativas independentes, cada
uma com seus erros em lugares diferentes. O início de cada frase passa a ser a
mediana de todas elas mais a do alinhador. Palavras perto da borda da janela e a
1ª palavra de cada trecho do Whisper ficam de fora (é onde ele mais erra).

Leva ~1 min por 15 s de áudio no CPU (modelo large-v3).
"""
import argparse
import json
import sys
import unicodedata
from difflib import SequenceMatcher

import numpy as np

from audio import TAXA, ler_audio

JANELA, PASSO, BORDA = 14.0, 3.5, 0.5
ANTECIPA = 0.20      # mesma antecipação do sincronizar.py
EMENDA = 0.30
MIN_ESTIMATIVAS = 2  # com menos votos que isso, vale o tempo do alinhador


def n(p):
    p = unicodedata.normalize("NFD", p.lower())
    return "".join(c for c in p if c.isalnum() and unicodedata.category(c) != "Mn")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("voz")
    p.add_argument("frases")
    p.add_argument("-o", "--saida", required=True)
    p.add_argument("--modelo", default="large-v3")
    a = p.parse_args()

    from faster_whisper import WhisperModel
    dados = json.load(open(a.frases, encoding="utf-8"))
    frases = dados["frases"]
    nossas = [(fi, k, n(w[0]), w[1]) for fi, f in enumerate(frases) for k, w in enumerate(f["palavras"])]
    votos = {(fi, k): [w2v] for fi, k, _, w2v in nossas}

    audio = ler_audio(a.voz)
    dur = len(audio) / TAXA
    modelo = WhisperModel(a.modelo, device="auto", compute_type="int8")
    inicios = np.arange(-PASSO, dur, PASSO)
    for o in inicios:
        o0, o1 = max(0.0, o), min(dur, o + JANELA)
        if o1 - o0 < 3:
            continue
        cand = [x for x in nossas if o0 - 2 <= x[3] <= o1 + 2]
        if not cand:
            continue
        dica = " ".join(dict.fromkeys(frases[x[0]]["texto"] for x in cand))
        trechos, _ = modelo.transcribe(audio[int(o0 * TAXA):int(o1 * TAXA)], language="pt", word_timestamps=True,
                                       beam_size=5, vad_filter=False, condition_on_previous_text=False,
                                       initial_prompt=dica)
        ouvidas = []
        for s in trechos:
            for j, w in enumerate(s.words or []):
                ini = o0 + w.start
                util = j > 0 and ini > o0 + BORDA and o0 + w.end < o1 - BORDA
                ouvidas.append((n(w.word), ini, util))
        sm = SequenceMatcher(None, [x[2] for x in cand], [x[0] for x in ouvidas], autojunk=False)
        for i, j, size in sm.get_matching_blocks():
            for d in range(size):
                if ouvidas[j + d][2]:
                    fi, k = cand[i + d][:2]
                    votos[(fi, k)].append(ouvidas[j + d][1])
        print(f"  janela {o0:6.1f}-{o1:6.1f}s: {len(ouvidas)} palavras ouvidas", file=sys.stderr)

    mudancas = []
    for fi, f in enumerate(frases):
        v = votos[(fi, 0)]
        w2v = v[0]
        if len(v) - 1 < MIN_ESTIMATIVAS:
            continue
        mediana = float(np.median(v))
        novo = mediana - ANTECIPA
        if fi > 0:
            novo = max(novo, frases[fi - 1]["inicio"] + 0.3)
        if abs(novo - f["inicio"]) >= 0.08:
            mudancas.append((fi + 1, f["texto"], f["inicio"], novo, len(v) - 1, np.std(v[1:])))
        f["inicio"] = round(novo, 3)
        f["votos"] = [round(x, 2) for x in v]

    for i, f in enumerate(frases):
        if i + 1 < len(frases):
            prox = frases[i + 1]["inicio"]
            if prox - f["fim"] < EMENDA or f["fim"] > prox:
                f["fim"] = prox
        f["fim"] = round(max(f["fim"], f["inicio"] + 0.4), 3)

    json.dump({"frases": frases, "sincronizado": True}, open(a.saida, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n{len(mudancas)} frases mudaram de início:", file=sys.stderr)
    for i, txt, de, para, nv, dp in mudancas:
        print(f"  {i:3d}  {de:7.2f} -> {para:7.2f}  ({nv} votos, dispersão {dp:.2f}s)  {txt}", file=sys.stderr)
    print(f"-> {a.saida}", file=sys.stderr)


if __name__ == "__main__":
    main()
