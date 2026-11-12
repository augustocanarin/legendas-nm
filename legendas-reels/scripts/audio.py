"""Leitura de áudio com o ffmpeg, compartilhada pelos scripts."""
import subprocess
import sys

import numpy as np

TAXA = 16000


def ler_audio(video, taxa=TAXA):
    """Extrai o áudio (mono, 16 kHz) como float32.

    Usamos o ffmpeg em vez do leitor interno do faster-whisper (PyAV) porque
    versões novas do PyAV quebram a compatibilidade; o ffmpeg já é obrigatório.
    """
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", video, "-vn", "-ac", "1",
                        "-ar", str(taxa), "-f", "s16le", "-"], capture_output=True)
    if r.returncode != 0:
        sys.exit("ffmpeg não conseguiu ler o áudio:\n" + r.stderr.decode(errors="ignore")[-2000:])
    return np.frombuffer(r.stdout, np.int16).astype(np.float32) / 32768.0
