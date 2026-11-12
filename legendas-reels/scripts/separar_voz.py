#!/usr/bin/env python3
"""Separa a voz dos instrumentos (para músicas/louvor).

Uso:
    python separar_voz.py VIDEO.mp4 -o trabalho/voz.wav

Com banda ou violão tocando, os modelos de transcrição e de alinhamento confundem
instrumento com voz e os tempos ficam imprecisos. A voz isolada resolve isso: use
o voz.wav no lugar do vídeo em transcrever.py e sincronizar.py. (A legenda é
gravada no vídeo original, com o áudio original — o voz.wav é só para análise.)

Modelo: Kim Vocal 2 (MDX-Net, Ultimate Vocal Remover), ~67 MB, em ONNX.
Leva cerca de 1x a duração do vídeo num computador comum.
"""
import argparse
import subprocess
import sys
import wave

import numpy as np

MODELO = ("diarizeapp/uvr-kim-vocal-2", "Kim_Vocal_2.onnx")
TAXA = 44100
N_FFT, HOP, DIM_F, DIM_T = 7680, 1024, 3072, 256
COMPENSA = 1.009
N_BINS = N_FFT // 2 + 1
CHUNK = HOP * (DIM_T - 1)
TRIM = N_FFT // 2
GERA = CHUNK - 2 * TRIM
JANELA = np.hanning(N_FFT + 1)[:-1].astype(np.float32)   # Hann periódica, igual à do treino


def ler_estereo(video):
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", video, "-vn", "-ac", "2", "-ar", str(TAXA),
                        "-f", "f32le", "-"], capture_output=True)
    if r.returncode != 0:
        sys.exit("ffmpeg não conseguiu ler o áudio:\n" + r.stderr.decode(errors="ignore")[-2000:])
    return np.frombuffer(r.stdout, np.float32).reshape(-1, 2).T.copy()


def stft(x):
    """[canais, amostras] -> [canais, N_BINS, quadros], igual ao torch.stft(center=True)."""
    x = np.pad(x, ((0, 0), (N_FFT // 2, N_FFT // 2)), mode="reflect")
    n = 1 + (x.shape[1] - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    return np.fft.rfft(x[:, idx] * JANELA, axis=-1).transpose(0, 2, 1)


def istft(espec, comprimento):
    quadros = np.fft.irfft(espec.transpose(0, 2, 1), n=N_FFT, axis=-1) * JANELA
    n = quadros.shape[1]
    total = N_FFT + HOP * (n - 1)
    saida = np.zeros((espec.shape[0], total), np.float32)
    norma = np.zeros(total, np.float32)
    for i in range(n):
        saida[:, i * HOP:i * HOP + N_FFT] += quadros[:, i]
        norma[i * HOP:i * HOP + N_FFT] += JANELA ** 2
    saida /= np.maximum(norma, 1e-8)
    return saida[:, N_FFT // 2:N_FFT // 2 + comprimento]


def separar(mix, sessao):
    n = mix.shape[1]
    pad = GERA - n % GERA
    mix_p = np.concatenate([np.zeros((2, TRIM), np.float32), mix,
                            np.zeros((2, pad + TRIM), np.float32)], axis=1)
    partes = []
    total = n + pad
    for i in range(0, total, GERA):
        pedaço = mix_p[:, i:i + CHUNK]
        S = stft(pedaço)[:, :DIM_F]                               # [2, F, T] complexo
        x = np.stack([S[0].real, S[0].imag, S[1].real, S[1].imag])[None].astype(np.float32)
        y = sessao.run(None, {"input": x})[0][0]
        V = np.zeros((2, N_BINS, DIM_T), np.complex64)
        V[0, :DIM_F] = y[0] + 1j * y[1]
        V[1, :DIM_F] = y[2] + 1j * y[3]
        partes.append(istft(V, CHUNK)[:, TRIM:-TRIM])
        print(f"  separando voz: {min(i + GERA, n) / TAXA:5.0f}s / {n / TAXA:.0f}s", file=sys.stderr, end="\r")
    print(file=sys.stderr)
    return np.concatenate(partes, axis=1)[:, :n] * COMPENSA


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("-o", "--saida", default="voz.wav")
    a = p.parse_args()
    try:
        import onnxruntime as ort
    except ImportError:
        sys.exit("Falta instalar: pip install onnxruntime")
    from huggingface_hub import hf_hub_download

    sessao = ort.InferenceSession(hf_hub_download(*MODELO), providers=["CPUExecutionProvider"])
    voz = separar(ler_estereo(a.video), sessao)
    pcm = (np.clip(voz.T, -1, 1) * 32767).astype("<i2")
    with wave.open(a.saida, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(TAXA)
        w.writeframes(pcm.tobytes())
    print(f"Voz isolada -> {a.saida}", file=sys.stderr)


if __name__ == "__main__":
    main()
