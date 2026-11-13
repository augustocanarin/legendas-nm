#!/usr/bin/env python3
"""Passo 3b: acerta o tempo de cada frase com precisão (alinhamento forçado).

Uso:
    python sincronizar.py VIDEO.mp4 frases.json -o frases_sincronizadas.json

O Whisper descobre O QUE foi dito, mas seus tempos por palavra erram — muito em
música (às vezes mais de um segundo). Este passo pega o texto JÁ REVISADO das
frases e procura no áudio, letra por letra, onde cada palavra começa e termina,
com um modelo wav2vec2 de português (o mesmo princípio de karaokê). Assim cada
frase entra quando a primeira sílaba soa e sai quando a última termina.

Rode sempre DEPOIS de revisar o texto: o alinhamento segue exatamente o que
está escrito, então uma palavra errada atrapalha o tempo das vizinhas.

Modelo: onnx-community/wav2vec2-large-xlsr-53-portuguese-ONNX (~1,2 GB, baixado
do huggingface.co na primeira vez). Precisa de: pip install onnxruntime
"""
import argparse
import json
import sys
import unicodedata

import numpy as np

from audio import TAXA, ler_audio

MODELO = "onnx-community/wav2vec2-large-xlsr-53-portuguese-ONNX"
QUADRO = 320 / TAXA          # o modelo dá uma previsão a cada 20 ms
FOLGA_BUSCA = 4.0            # quanto (s) a frase pode estar longe do tempo aproximado
BLOCO, CONTEXTO = 20.0, 1.0  # o áudio é processado em blocos de 20 s com 1 s de contexto

ANTECIPA = 0.20              # a frase aparece 0,2 s antes da sílaba: assim o olho a percebe "junto" com a voz
COLADA = 0.60                # frases separadas por menos que isso são continuação do mesmo verso
EMENDA = 0.30                # buracos menores que isso entre frases são fechados (evita piscar)
SUSTAIN_LIMITE = 6.0         # máximo que uma nota sustentada pode estender a frase
VOZ_ACIMA_DO_PISO_DB = 12    # há voz quando a energia está 12 dB acima do silêncio do arquivo
PAUSA_TOLERADA = 0.25        # respiros/vibratos mais curtos que isso não encerram a frase


def carregar_modelo():
    try:
        import onnxruntime as ort
    except ImportError:
        sys.exit("Falta instalar: pip install onnxruntime")
    from huggingface_hub import hf_hub_download
    vocab = json.load(open(hf_hub_download(MODELO, "vocab.json"), encoding="utf-8"))
    opcoes = ort.SessionOptions()
    opcoes.log_severity_level = 3
    sessao = ort.InferenceSession(hf_hub_download(MODELO, "onnx/model.onnx"), opcoes,
                                  providers=["CPUExecutionProvider"])
    return sessao, vocab


def emissoes(sessao, audio):
    """Log-probabilidades por quadro (20 ms) para o áudio inteiro, processado em blocos."""
    passo, ctx = int(BLOCO * TAXA), int(CONTEXTO * TAXA)
    total = int(np.ceil(len(audio) / 320))
    partes = []
    for ini in range(0, len(audio), passo):
        a, b = max(0, ini - ctx), min(len(audio), ini + passo + ctx)
        x = audio[a:b]
        x = (x - x.mean()) / (x.std() + 1e-7)
        logits = sessao.run(None, {"input_values": x[None, :].astype(np.float32)})[0][0]
        q0 = (ini - a) // 320
        q1 = q0 + min(passo, len(audio) - ini) // 320
        partes.append(logits[q0:q1])
        print(f"  áudio processado: {min(ini + passo, len(audio)) / TAXA:5.0f}s", file=sys.stderr, end="\r")
    print(file=sys.stderr)
    lg = np.concatenate(partes)[:total]
    lg = lg - lg.max(axis=1, keepdims=True)
    return lg - np.log(np.exp(lg).sum(axis=1, keepdims=True))


def para_tokens(palavra, vocab):
    """Letras da palavra no alfabeto do modelo (sem pontuação; acento desconhecido vira letra base)."""
    ids = []
    for c in palavra.lower():
        if c in vocab and c not in "|'-":
            ids.append(vocab[c])
        else:
            base = unicodedata.normalize("NFD", c)[0]
            if base in vocab and base.isalpha():
                ids.append(vocab[base])
            elif c.isdigit():
                ids.append(vocab["<unk>"])
    return ids


def montar_sequencia(frases, vocab):
    """Lista de tokens com, para cada um, a frase e a palavra a que pertence."""
    tokens, frase_de, palavra_de, palavras = [], [], [], []
    sep = vocab["|"]
    for fi, f in enumerate(frases):
        for texto in f["texto"].split():
            ids = para_tokens(texto, vocab)
            if not ids:
                continue
            if tokens:
                tokens.append(sep); frase_de.append(fi); palavra_de.append(-1)
            for t in ids:
                tokens.append(t); frase_de.append(fi); palavra_de.append(len(palavras))
            palavras.append({"palavra": texto, "frase": fi})
    return np.array(tokens), np.array(frase_de), np.array(palavra_de), palavras


def alinhar(lp, tokens, permitido):
    """Viterbi CTC: escolhe, quadro a quadro, qual letra está soando.

    `permitido[s]` = (primeiro, último) quadro em que o token s pode aparecer —
    impede que uma frase "escorregue" para longe de onde o Whisper a ouviu.
    """
    T, L = lp.shape[0], len(tokens)
    S = 2 * L + 1
    estado_tok = np.full(S, -1)
    estado_tok[1::2] = np.arange(L)
    branco = 0  # <pad> é o "branco" do CTC neste modelo
    ids = np.where(estado_tok >= 0, tokens[np.maximum(estado_tok, 0)], branco)
    pode_pular = np.zeros(S, bool)
    pode_pular[3::2] = tokens[1:] != tokens[:-1]
    ini = np.zeros(S); fim = np.full(S, T - 1)
    ini[1::2], fim[1::2] = permitido[:, 0], permitido[:, 1]

    NEG = -1e9
    alfa = np.full(S, NEG)
    alfa[0], alfa[1] = lp[0, branco], (lp[0, ids[1]] if ini[1] <= 0 else NEG)
    volta = np.zeros((T, S), np.int8)
    for t in range(1, T):
        c0 = alfa
        c1 = np.concatenate(([NEG], alfa[:-1]))
        c2 = np.where(pode_pular, np.concatenate(([NEG, NEG], alfa[:-2])), NEG)
        pilha = np.stack([c0, c1, c2])
        esc = pilha.argmax(0)
        alfa = pilha[esc, np.arange(S)] + lp[t, ids]
        fora = (t < ini) | (t > fim)
        alfa[fora] = NEG
        volta[t] = esc
    s = S - 1 if alfa[S - 1] >= alfa[S - 2] else S - 2
    if alfa[s] <= NEG / 2:
        return None
    caminho = np.empty(T, int)
    for t in range(T - 1, -1, -1):
        caminho[t] = s
        s -= int(volta[t, s])
    return caminho, estado_tok


def energia_voz(audio):
    """Energia (RMS) da faixa de voz por quadro de 20 ms, suavizada."""
    n = len(audio) // 320
    x = audio[: n * 320].reshape(n, 320)
    rms = np.sqrt((x ** 2).mean(axis=1) + 1e-10)
    k = np.ones(5) / 5
    return np.convolve(rms, k, mode="same")


def voz_ativa(rms):
    """Quadros com voz soando: energia bem acima do piso de silêncio do arquivo.

    Funciona de verdade só com a voz isolada (separar_voz.py); no áudio com
    instrumentos, o violão também conta como "som".
    """
    db = 20 * np.log10(rms + 1e-9)
    piso = np.percentile(db, 10)
    return db > piso + VOZ_ACIMA_DO_PISO_DB


def estender_sustain(fim_q, limite_q, ativa):
    """Mantém a frase na tela enquanto a última nota continua soando.

    O alinhamento marca o fim da última letra cedo (o modelo "reconhece" a
    vogal no início e não enquanto ela é segurada), então seguimos a energia
    da voz até ela realmente parar.
    """
    tolera = int(PAUSA_TOLERADA / QUADRO)
    q, silencio = fim_q, 0
    while q + 1 < limite_q and silencio <= tolera:
        q += 1
        silencio = 0 if ativa[q] else silencio + 1
    return max(fim_q, q - silencio)


def _n(p):
    p = unicodedata.normalize("NFD", p.lower())
    return "".join(c for c in p if c.isalnum() and unicodedata.category(c) != "Mn")


def casar_whisper(novas, transcricao):
    """Para cada frase, os tempos que o Whisper deu à 1ª e à última palavra (ou None).

    A 1ª palavra de cada trecho do Whisper é ignorada: nela ele costuma "colar" o
    início no fim do trecho anterior, errando por segundos.
    """
    from difflib import SequenceMatcher
    W = transcricao["palavras"]
    inicios_seg = {round(sg["inicio"], 2) for sg in transcricao.get("segmentos", [])}
    confiavel = [round(w["inicio"], 2) not in inicios_seg for w in W]
    nossas = [(fi, k, _n(p[0])) for fi, f in enumerate(novas) for k, p in enumerate(f["palavras"])]
    sm = SequenceMatcher(None, [x[2] for x in nossas], [_n(w["palavra"]) for w in W], autojunk=False)
    par = {}
    for a, b, size in sm.get_matching_blocks():
        for d in range(size):
            par[nossas[a + d][:2]] = b + d
    res = []
    for fi, f in enumerate(novas):
        j0, j1 = par.get((fi, 0)), par.get((fi, len(f["palavras"]) - 1))
        ini = W[j0]["inicio"] if j0 is not None and confiavel[j0] else None
        fim = W[j1]["fim"] if j1 is not None else None
        res.append((ini, fim))
    return res


def fundir_com_whisper(novas, tempos_w):
    """Combina os dois relógios onde o alinhador costuma errar.

    - Troca de frase dentro do mesmo verso (sem pausa): o alinhador marca a
      entrada da 2ª parte ~0,3-0,5 s atrasada; usamos a média dos dois.
    - Palavras espremidas (frase com menos de 0,3 s de voz, típico de nota
      longa): o alinhador se perdeu ali, então vale o tempo do Whisper.
    """
    ajustes = []
    seguir_whisper = False   # depois de um trecho espremido, a frase seguinte também está deslocada
    for i, (f, (wi, wf)) in enumerate(zip(novas, tempos_w)):
        if wi is None:
            seguir_whisper = False
            continue
        w2v_ini = f["palavras"][0][1]
        espremida = f["palavras"][-1][2] - w2v_ini < 0.3
        colada = i > 0 and w2v_ini - novas[i - 1]["palavras"][-1][2] < COLADA
        if (espremida or seguir_whisper) and abs(wi - w2v_ini) < FOLGA_BUSCA:
            novo = wi
            seguir_whisper = espremida
        elif colada and abs(wi - w2v_ini) < 1.0:
            novo = (wi + w2v_ini) / 2
            seguir_whisper = False
        else:
            seguir_whisper = False
            continue
        if i > 0:
            novo = max(novo, novas[i - 1]["inicio"] + 0.3)
        if i + 1 < len(novas) and not espremida:
            novo = min(novo, novas[i + 1]["inicio"] - 0.3)
        if abs(novo - f["inicio"]) >= 0.05:
            ajustes.append((i + 1, f["texto"], f["inicio"], novo))
            f["inicio"] = novo
            if espremida and wf is not None and wf > novo:
                f["fim"] = max(f["fim"], wf)
    return ajustes


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("frases")
    p.add_argument("-o", "--saida", default=None)
    p.add_argument("--transcricao", default=None,
                   help="transcricao.json do Whisper: combina os dois relógios (recomendado)")
    p.add_argument("--sem-sustain", action="store_true",
                   help="não estender frases durante notas longas (use em vídeos só de fala)")
    a = p.parse_args()

    dados = json.load(open(a.frases, encoding="utf-8"))
    frases = sorted(dados["frases"], key=lambda f: f["inicio"])
    audio = ler_audio(a.video)
    print("Carregando o modelo de alinhamento...", file=sys.stderr)
    sessao, vocab = carregar_modelo()
    lp = emissoes(sessao, audio)
    T = lp.shape[0]

    tokens, frase_de, palavra_de, palavras = montar_sequencia(frases, vocab)
    folga = int(FOLGA_BUSCA / QUADRO)
    janelas = np.array([[max(0, int(f["inicio"] / QUADRO) - folga), min(T - 1, int(f["fim"] / QUADRO) + folga)]
                        for f in frases])
    permitido = janelas[frase_de]
    print(f"Alinhando {len(palavras)} palavras em {T * QUADRO:.0f}s de áudio...", file=sys.stderr)
    r = alinhar(lp, tokens, permitido)
    if r is None:
        sys.exit("Não consegui alinhar: o texto das frases provavelmente não corresponde ao áudio "
                 "(trecho faltando ou sobrando). Revise o frases.json.")
    caminho, estado_tok = r

    # quadros de cada palavra
    q_ini = np.full(len(palavras), -1); q_fim = np.full(len(palavras), -1)
    for t, s in enumerate(caminho):
        k = estado_tok[s]
        if k >= 0 and palavra_de[k] >= 0:
            w = palavra_de[k]
            if q_ini[w] < 0:
                q_ini[w] = t
            q_fim[w] = t
    for w, pal in enumerate(palavras):
        pal["inicio"], pal["fim"] = q_ini[w] * QUADRO, (q_fim[w] + 1) * QUADRO

    rms = energia_voz(audio)[:T]
    ativa = voz_ativa(rms)

    novas = []
    for fi, f in enumerate(frases):
        ws = [w for w in palavras if w["frase"] == fi]
        if not ws:
            continue
        inicio, fim = ws[0]["inicio"], ws[-1]["fim"]
        novas.append({"inicio": inicio, "fim": fim, "texto": f["texto"],
                      "palavras": [[w["palavra"], round(w["inicio"], 3), round(w["fim"], 3)] for w in ws]})

    if a.transcricao:
        tw = casar_whisper(novas, json.load(open(a.transcricao, encoding="utf-8")))
        ajustes = fundir_com_whisper(novas, tw)
        if ajustes:
            print(f"Ajustadas com o Whisper ({len(ajustes)}):", file=sys.stderr)
            for n, txt, de, para in ajustes:
                print(f"  {n:3d}  {de:7.2f} -> {para:7.2f}  {txt}", file=sys.stderr)
        for i in range(len(novas) - 1):          # a frase anterior termina onde a próxima começa, se passar
            novas[i]["fim"] = min(novas[i]["fim"], novas[i + 1]["inicio"])

    for i, f in enumerate(novas):
        prox = novas[i + 1]["inicio"] if i + 1 < len(novas) else T * QUADRO
        if not a.sem_sustain:
            limite = min(prox, f["fim"] + SUSTAIN_LIMITE)
            f["fim"] = (estender_sustain(int(f["fim"] / QUADRO), int(limite / QUADRO), ativa) + 1) * QUADRO
        f["inicio"] = max(0.0, f["inicio"] - ANTECIPA)
    for i, f in enumerate(novas):
        if i + 1 < len(novas):
            prox = novas[i + 1]["inicio"]
            if prox - f["fim"] < EMENDA:
                f["fim"] = prox
            f["fim"] = min(f["fim"], prox)
        f["inicio"], f["fim"] = round(f["inicio"], 3), round(f["fim"], 3)

    saida = a.saida or a.frases.replace(".json", "_sincronizadas.json")
    json.dump({"frases": novas, "sincronizado": True}, open(saida, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    for i, (f, velha) in enumerate(zip(novas, frases), 1):
        d = f["inicio"] - velha["inicio"]
        print(f"{i:3d}  {f['inicio']:7.2f}-{f['fim']:7.2f}  (antes {velha['inicio']:7.2f}, {d:+.2f}s)  {f['texto']}",
              file=sys.stderr)
    curtas = [(i, f) for i, f in enumerate(novas, 1) if f["fim"] - f["inicio"] < 0.5]
    if curtas:
        print("\nATENÇÃO: frases que só piscam na tela (alinhamento provavelmente espremido ali —"
              " compare com os tempos do transcricao.json e ajuste à mão):", file=sys.stderr)
        for i, f in curtas:
            print(f"  {i:3d}  {f['inicio']:7.2f}-{f['fim']:7.2f}  {f['texto']}", file=sys.stderr)
    print(f"\n{len(novas)} frases sincronizadas -> {saida}", file=sys.stderr)


if __name__ == "__main__":
    main()
