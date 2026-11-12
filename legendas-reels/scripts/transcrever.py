#!/usr/bin/env python3
"""Passo 1: transcreve a fala do vídeo com o tempo de cada palavra.

Uso:
    python transcrever.py VIDEO.mp4 -o transcricao.json [--modelo medium] [--dica "nomes próprios"]

Gera um JSON com a lista de palavras ({"palavra", "inicio", "fim", "confianca"})
e o texto corrido de cada trecho, que o segmentar.py usa em seguida.
"""
import argparse
import json
import sys
import time

from audio import ler_audio

DICA_PADRAO = (
    "Ministração cristã em português do Brasil, com versículos e louvor. "
    "Jesus, Cristo, Deus, Senhor, Espírito Santo, Pai, aleluia, amém, glória, "
    "graça, louvor, adoração, Bíblia, evangelho, igreja, Getsêmani, Emanuel, Calvário."
)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("-o", "--saida", default="transcricao.json")
    p.add_argument("--modelo", default="medium",
                   help="tiny | base | small | medium | large-v3 (maior = mais preciso e mais lento)")
    p.add_argument("--dica", default="",
                   help="nomes e termos que aparecem no vídeo (pregador, música, livro bíblico)")
    p.add_argument("--idioma", default="pt")
    p.add_argument("--sem-vad", action="store_true",
                   help="não pular trechos sem voz (use se partes cantadas sumirem da transcrição)")
    a = p.parse_args()

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("Falta instalar: pip install faster-whisper")

    dica = (DICA_PADRAO + " " + a.dica).strip()
    t0 = time.time()
    print(f"Carregando modelo '{a.modelo}' (na 1ª vez ele é baixado, ~1,5 GB no medium)...", file=sys.stderr)
    modelo = WhisperModel(a.modelo, device="auto", compute_type="int8")
    trechos, info = modelo.transcribe(
        ler_audio(a.video),
        language=a.idioma,
        word_timestamps=True,
        initial_prompt=dica,
        vad_filter=not a.sem_vad,
        vad_parameters={"min_silence_duration_ms": 400},
        beam_size=5,
    )

    palavras, segmentos = [], []
    for s in trechos:
        segmentos.append({"inicio": round(s.start, 3), "fim": round(s.end, 3), "texto": s.text.strip()})
        for w in s.words or []:
            texto = w.word.strip()
            if texto:
                palavras.append({"palavra": texto, "inicio": round(w.start, 3),
                                 "fim": round(w.end, 3), "confianca": round(w.probability, 3)})
        print(f"  [{s.start:6.2f}s] {s.text.strip()}", file=sys.stderr)

    saida = {"video": a.video, "duracao": round(info.duration, 3), "modelo": a.modelo,
             "palavras": palavras, "segmentos": segmentos}
    with open(a.saida, "w", encoding="utf-8") as f:
        json.dump(saida, f, ensure_ascii=False, indent=1)

    duvidosas = [w["palavra"] for w in palavras if w["confianca"] < 0.5]
    print(f"\n{len(palavras)} palavras em {time.time() - t0:.0f}s -> {a.saida}", file=sys.stderr)
    if duvidosas:
        print(f"Palavras com baixa confiança (revise): {', '.join(duvidosas[:40])}", file=sys.stderr)


if __name__ == "__main__":
    main()
