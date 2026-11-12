#!/usr/bin/env python3
"""Passo 3: grava as frases revisadas no vídeo, no estilo da referência.

Uso:
    python legendar.py VIDEO.mp4 frases.json -o VIDEO_legendado.mp4
        [--previa]                      só gera imagens de conferência, sem renderizar o vídeo
        [--fonte ARQ] [--fonte-nome N]  usa outra fonte (ex.: Helvetica Neue do Mac)
        [--estilo estilo.json]

Também salva, ao lado do vídeo final, o .srt (legenda em arquivo) e o .ass
(legenda já estilizada, que abre no Aegisub/VLC para ajustes finos).
"""
import argparse
import os
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from estilo import Medidor, aplicar_fonte, carregar_estilo, escala, info_fonte, preparar_texto


def dimensoes(video):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height:stream_side_data=rotation:stream_tags=rotate",
                        "-of", "json", video], capture_output=True, text=True, check=True)
    s = json.loads(r.stdout)["streams"][0]
    w, h = s["width"], s["height"]
    rot = 0
    for sd in s.get("side_data_list", []):
        rot = int(float(sd.get("rotation", 0) or 0))
    rot = rot or int(s.get("tags", {}).get("rotate", 0) or 0)
    if abs(rot) % 180 == 90:
        w, h = h, w  # o ffmpeg gira o vídeo automaticamente; a legenda segue a imagem final
    return w, h


def tem_audio(video):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                        "stream=codec_name", "-of", "csv=p=0", video], capture_output=True, text=True)
    return r.stdout.strip().splitlines()[0] if r.stdout.strip() else None


def limpar(texto):
    texto = " ".join(texto.split())
    texto = re.sub(r"[,.;:…]+(?=[\"”')]*$)", "", texto)   # tira pontuação final, mantém ? e !
    return texto.strip()


def cor_ass(hexa, alpha=0):
    hexa = hexa.lstrip("#")
    r, g, b = hexa[0:2], hexa[2:4], hexa[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def tempo_ass(t):
    t = max(0.0, t)
    cs = int(round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def tempo_srt(t):
    ms = int(round(max(0.0, t) * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def escapar_ass(t):
    return t.replace("\\", "\\\\").replace("{", "(").replace("}", ")")


def montar_ass(frases, estilo, w, h):
    k = escala(estilo, w)
    fonte = info_fonte(estilo)
    medidor = Medidor(estilo, w)
    tamanho = estilo["tamanho_px"] * k * fonte["fator_ass"]
    espaco = estilo.get("espacamento_letras_px", 0) * k
    sx = estilo.get("escala_horizontal", 100)
    brilho = estilo.get("brilho_desfoque_px", 0) * k
    sombra = estilo.get("sombra_contraste", 0) * k
    cor = cor_ass(estilo["cor"])
    x, y = w / 2, h * estilo.get("posicao_vertical", 0.5)

    linhas = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {w}", f"PlayResY: {h}",
        "WrapStyle: 2", "ScaledBorderAndShadow: yes", "YCbCr Matrix: TV.709", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Legenda,{fonte['nome']},{tamanho:.2f},{cor},{cor},&H00000000,&H00000000,"
        f"0,0,0,0,{sx},100,{espaco:.2f},0,1,0,0,5,0,0,0,1",
        "", "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    ajustadas = []
    for i, f in enumerate(frases, 1):
        texto = preparar_texto(estilo, limpar(f["texto"]))
        if not texto:
            continue
        largura = medidor.largura(texto)
        zoom = ""
        if largura > medidor.limite:  # frase editada ficou longa: encolhe só ela para caber numa linha
            pct = 100 * medidor.limite / largura
            zoom = f"\\fscx{sx * pct / 100:.1f}\\fscy{pct:.1f}"
            ajustadas.append((i, texto, round(pct)))
        t0, t1 = tempo_ass(f["inicio"]), tempo_ass(f["fim"])
        base = f"{{\\an5\\pos({x:.1f},{y:.1f}){zoom}"
        txt = escapar_ass(texto)
        if sombra:
            linhas.append(f"Dialogue: 0,{t0},{t1},Legenda,,0,0,0,,{base}\\1c&H000000&\\1a&H50&\\blur{sombra:.1f}}}{txt}")
        linhas.append(f"Dialogue: 1,{t0},{t1},Legenda,,0,0,0,,{base}}}{txt}")
        if brilho:
            # cópia desfocada por cima, igual à camada de blur do Figma
            linhas.append(f"Dialogue: 2,{t0},{t1},Legenda,,0,0,0,,{base}\\blur{brilho:.1f}}}{txt}")
    return "\n".join(linhas) + "\n", ajustadas


def montar_srt(frases):
    blocos = []
    for i, f in enumerate([f for f in frases if limpar(f["texto"])], 1):
        blocos.append(f"{i}\n{tempo_srt(f['inicio'])} --> {tempo_srt(f['fim'])}\n{limpar(f['texto'])}\n")
    return "\n".join(blocos)


def validar(frases):
    problemas = []
    for i, f in enumerate(frases, 1):
        if f["fim"] <= f["inicio"]:
            problemas.append(f"frase {i}: fim ({f['fim']}) não é depois do início ({f['inicio']})")
        if i < len(frases) and f["fim"] > frases[i]["inicio"] + 0.01:
            problemas.append(f"frase {i} invade a frase {i + 1} (duas frases ao mesmo tempo na tela)")
    return problemas


def rodar_ffmpeg(args):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"] + args,
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("ffmpeg falhou:\n" + r.stderr[-3000:])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("frases")
    p.add_argument("-o", "--saida", default=None)
    p.add_argument("--estilo", default=None)
    p.add_argument("--fonte", default=None)
    p.add_argument("--fonte-nome", default=None)
    p.add_argument("--previa", action="store_true", help="gera PNGs de algumas frases em vez do vídeo")
    p.add_argument("--crf", type=int, default=18, help="qualidade do vídeo (menor = melhor; 18 é quase sem perda)")
    a = p.parse_args()

    video = str(Path(a.video).resolve())
    estilo = carregar_estilo(a.estilo)
    if a.fonte:
        aplicar_fonte(estilo, a.fonte, a.fonte_nome)
    with open(a.frases, encoding="utf-8") as f:
        frases = json.load(f)["frases"]
    frases.sort(key=lambda f: f["inicio"])
    problemas = validar(frases)
    if problemas:
        sys.exit("Corrija o frases.json antes de legendar:\n  " + "\n  ".join(problemas))

    w, h = dimensoes(video)
    ass, ajustadas = montar_ass(frases, estilo, w, h)
    saida = Path(a.saida or Path(video).with_name(Path(video).stem + "_legendado.mp4")).resolve()
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.with_suffix(".ass").write_text(ass, encoding="utf-8")
    saida.with_suffix(".srt").write_text(montar_srt(frases), encoding="utf-8")
    for i, texto, pct in ajustadas:
        dica = " Ficou pequena demais: divida esta frase em duas no frases.json." if pct < 80 else ""
        print(f"Aviso: frase {i} ('{texto}') não cabia numa linha; foi reduzida para {pct}% do tamanho.{dica}",
              file=sys.stderr)

    with tempfile.TemporaryDirectory() as tmp:
        # trabalhar numa pasta temporária evita problemas de caminho com espaço/acento no filtro do ffmpeg
        shutil.copy(estilo["fonte_arquivo"], Path(tmp) / Path(estilo["fonte_arquivo"]).name)
        (Path(tmp) / "legenda.ass").write_text(ass, encoding="utf-8")
        filtro = "subtitles=legenda.ass:fontsdir=."
        os.chdir(tmp)

        if a.previa:
            pasta = saida.with_name(saida.stem + "_previa")
            pasta.mkdir(exist_ok=True)
            escolhidas = sorted({0, len(frases) // 2, len(frases) - 1})
            for n in escolhidas:
                f = frases[n]
                t = (f["inicio"] + f["fim"]) / 2
                png = pasta / f"frase_{n + 1:02d}.png"
                rodar_ffmpeg(["-i", video, "-vf", f"{filtro},select='gte(t\\,{t:.3f})'",
                              "-frames:v", "1", "-update", "1", str(png)])
                print(f"Prévia: {png}  ({f['texto']})")
            return

        audio = tem_audio(video)
        args = ["-i", video, "-vf", filtro, "-c:v", "libx264", "-preset", "medium", "-crf", str(a.crf),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        if audio in ("aac", "mp3"):
            args += ["-c:a", "copy"]
        elif audio:
            args += ["-c:a", "aac", "-b:a", "192k"]
        rodar_ffmpeg(args + [str(saida)])
    print(f"Vídeo legendado: {saida}")
    print(f"Legenda em arquivo: {saida.with_suffix('.srt')}")


if __name__ == "__main__":
    main()
