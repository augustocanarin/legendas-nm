---
name: legendas-reels
description: Legenda Reels/vídeos verticais da igreja e do ministério de louvor gravando a fala na tela, uma frase por vez, centralizada, no estilo da casa (Helvetica Neue Condensed Bold amarela #FFEA00 com brilho). Transcreve o áudio em português com tempo por palavra, corrige nomes bíblicos e termos cristãos, quebra em frases de uma linha e entrega o .mp4 legendado + .srt. Use sempre que o usuário enviar ou mencionar um vídeo e pedir para legendar, colocar legenda, transcrever na tela, "legendar o reels", "pôr as falas no vídeo", gerar .srt de uma pregação, ministração, devocional, culto, louvor ou corte para Instagram/TikTok/Shorts — mesmo que não diga "skill" ou "estilo".
---

# Legendas para Reels (igreja / louvor)

Esta skill transforma um vídeo vertical em um vídeo com a fala legendada no estilo
definido no Figma: **uma frase por vez, numa linha só, no centro da tela,
caixa alta, amarelo #FFEA00, Helvetica Neue Condensed Bold, com um brilho suave**
(uma cópia desfocada do texto por cima). O estilo está em `assets/estilo.json`.

O público vê o Reel muitas vezes sem som, então a legenda é o que carrega a mensagem.
Uma palavra errada num versículo ou num nome bíblico chama mais atenção do que qualquer
efeito visual — por isso a revisão do texto (passo 3) é a parte mais importante.

## Preparar o ambiente (uma vez por máquina)

```bash
pip install faster-whisper pillow fonttools onnxruntime
ffmpeg -version   # precisa existir; no Mac: brew install ffmpeg
```

Na primeira transcrição, o faster-whisper baixa o modelo de voz do site huggingface.co
(~1,5 GB no `medium`). Se o download for bloqueado pela rede, avise o usuário: é preciso
liberar `huggingface.co` e os servidores de download dele (`us.aws.cdn.hf.co`, `cdn-lfs.hf.co`,
`cas-bridge.xethub.hf.co`) — em sessões na nuvem, nas configurações do ambiente.

Os scripts ficam em `scripts/` desta skill (nos comandos abaixo, `scripts/` é relativo à pasta
da skill; use o caminho completo). Salve os arquivos de trabalho numa pasta própria do vídeo,
nunca dentro da skill.

## Fluxo

### 1. Receber e entender o vídeo
Vídeos de Reels costumam passar do limite de anexo da conversa (~30 MB). Nesse caso, peça um
link do Google Drive com acesso "Qualquer pessoa com o link" e baixe direto (não use o conector
do Drive para vídeos: ele devolve o arquivo em base64 dentro da conversa e não cabe):
```bash
curl -sSL -o entrada/video.mp4 "https://drive.usercontent.google.com/download?id=<ID>&export=download&confirm=t"
```
(o `<ID>` é o trecho entre `/d/` e `/view` do link; em sessões na nuvem, `drive.google.com` e
`drive.usercontent.google.com` precisam estar liberados na rede.)

Confirme o arquivo e pergunte só o que mudar o resultado e não for óbvio:
- Quem fala e quais nomes próprios aparecem (pregador, igreja, música, livro/versículo citado).
  Isso vira a `--dica` da transcrição e melhora muito a precisão.
- Se há trechos **cantados**: legende a letra cantada também. Peça a **letra** (nome da música
  ajuda). Sites de letra como o Letras.mus.br bloqueiam acesso de servidores, então peça para o
  usuário colar o texto na conversa. Com a letra, siga o passo 2b.

### 2. Transcrever
```bash
python scripts/transcrever.py VIDEO.mp4 -o trabalho/transcricao.json --dica "Pr. Fulano, Filipenses 3, Getsêmani"
```
Modelo padrão `medium` (bom equilíbrio para fala em português); `small` se a máquina for fraca.
O script lista as palavras de baixa confiança — elas são o primeiro lugar a revisar.

**Música / louvor:** primeiro isole a voz (o violão/banda atrapalha tanto a transcrição quanto
o tempo), e use o `voz.wav` no lugar do vídeo na transcrição e na sincronização:
```bash
python scripts/separar_voz.py VIDEO.mp4 -o trabalho/voz.wav          # ~1x a duração do vídeo
python scripts/transcrever.py trabalho/voz.wav -o trabalho/transcricao.json --modelo large-v3 --sem-vad
```
Use `--modelo large-v3 --sem-vad` em música.

Sem o filtro de voz, o Whisper às vezes **inventa texto no fim do arquivo** (no teste, repetiu
o começo da música duas vezes nos últimos 3 s). Sinais: vários versos em poucos segundos,
palavras com duração zero e confiança perto de 0 no `transcricao.json`. Apague esses trechos
das frases antes de sincronizar. Também confira se o cantor repete ou omite palavras em
relação à letra (ex.: "outra vez, outra vez, e outra vez"): a legenda segue o que é
**cantado**, não a letra escrita — o usuário percebe quando aparece palavra que não foi cantada. No teste com um louvor de 3 min
(voz e violão), o `medium` com o filtro de voz pegou só 60 palavras e pulou metade da música;
o `large-v3` sem filtro pegou a letra inteira com bons tempos (~4 min de processamento).

### 2b. Músicas: aplicar a letra oficial
```bash
python scripts/alinhar_letra.py trabalho/transcricao.json trabalho/letra.txt -o trabalho/transcricao_letra.json
```
Salve a letra em `letra.txt`, um verso por linha, como o usuário mandou (refrão escrito uma
vez só já basta — cada repetição cantada é reconhecida). O script põe o texto da letra nos
tempos do áudio e marca o fim de cada verso como quebra obrigatória de frase. Confira a
lista impressa: a nota entre parênteses é a semelhança (abaixo de ~0,7 merece um olhar), e os
"não encaixados" são falas, "ô/oh" ou improvisos — decida com o usuário se ficam. Se ele
removeu algo da letra que mandou (ex.: um "ô"), isso indica que não quer na legenda. Use o
arquivo gerado no passo 3 no lugar da transcrição.

### 3. Quebrar em frases e revisar o texto
```bash
python scripts/segmentar.py trabalho/transcricao.json -o trabalho/frases.json
```
O script já garante frases curtas (2–3 palavras, no ritmo da referência "TUDO É PERDA") e evita cortes feios. Se você juntar/editar frases à mão e elas ficarem longas, rode o `segmentar.py` de novo passando o próprio `frases.json` revisado: ele redivide mantendo o texto e os limites que você definiu. Agora **leia
todas as frases** e corrija o `"texto"` no `frases.json`:
- Ortografia, acentos e pontuação; nomes bíblicos e termos cristãos — veja
  `references/vocabulario.md` (erros comuns do Whisper e grafia certa).
- Versículos: se a pessoa cita um texto bíblico, confira contra a versão que ela parece usar
  (NVI, ARA, NVT...) só para corrigir palavras mal ouvidas — não troque o que ela realmente falou.
- Pronomes referentes a Deus/Jesus seguem o uso do pregador; na dúvida, não force maiúscula
  (a legenda é toda em caixa alta mesmo).
- Vícios de fala ("né", "tipo", "éé") e repetições gaguejadas podem sair, desde que o sentido
  fique igual. Não reescreva a mensagem.
- Para **dividir** uma frase, use os tempos em `"palavras"` (cada item é `[palavra, início, fim]`)
  e crie duas entradas. Para **juntar**, use o início da primeira e o fim da segunda. Frases
  nunca podem se sobrepor no tempo.

### 3b. Sincronizar (obrigatório)
```bash
python scripts/sincronizar.py trabalho/voz.wav trabalho/frases.json --transcricao trabalho/transcricao.json -o trabalho/frases_sinc.json   # música
python scripts/sincronizar.py VIDEO.mp4 trabalho/frases.json -o trabalho/frases_sinc.json --sem-sustain   # fala
```
Os tempos do Whisper erram muito em música (no teste, até 3,4 s: ele "gruda" o início de um
verso no fim do anterior). O usuário espera que cada frase entre e saia exatamente quando é
cantada/falada — esse é o critério de qualidade nº 1. Este passo faz alinhamento forçado
(wav2vec2 de português) do texto já revisado contra o áudio. Por isso rode-o **depois** de
revisar o texto, e use `frases_sinc.json` no passo 5. Se ele disser que não conseguiu
alinhar, há texto faltando ou sobrando em relação ao áudio.

Passe sempre `--transcricao` em música. O alinhador acerta as trocas de frase com precisão
(elas caem exatamente nas quedas de volume entre sílabas) — não "corrija" isso com os tempos do
Whisper: testamos a média dos dois e o refrão piorou. O erro real dele é outro: em nota longa,
a 1ª palavra da frase seguinte fica presa antes da **respiração** do cantor ("caminho… | Ele
abriu no mar" com o "Ele" dentro do "caminhooo"). O script detecta as respirações na voz
isolada e move essas frases para quando a voz volta (lista o que mudou); a transcrição só
desempata. Depois de uma correção dessas, confira a frase SEGUINTE: ela também pode ter sido
espremida (compare com outra repetição do mesmo verso, que costuma ter o mesmo intervalo).
As legendas entram 0,2 s antes da sílaba, de propósito. Se o script avisar de frases que
"só piscam", confira se o trecho é mesmo cantado rápido antes de mexer.

Mostre ao usuário a lista final de frases (número, tempo e texto) e as correções que você fez
e que mudam sentido ou nome — ele conhece o conteúdo e pega erros que você não pegaria. Em
pedidos do tipo "só legenda e me manda", siga em frente e liste as correções no final.

### 4. Conferir o visual antes de renderizar
```bash
python scripts/legendar.py VIDEO.mp4 trabalho/frases_sinc.json -o saida/VIDEO_legendado.mp4 --previa
```
Isso gera 3 imagens (início, meio, fim). Olhe-as:
- A legenda está legível? O estilo original é amarelo sem contorno, pensado para fundo escuro.
  Se o fundo for claro (palco iluminado, parede branca, céu), sugira ativar a sombra de
  contraste: copie `assets/estilo.json`, mude `"sombra_contraste"` para `12` e passe
  `--estilo` com a cópia. Não altere o estilo da casa sem o usuário concordar.
- O centro da tela cobre um rosto o tempo todo? Então ofereça `"posicao_vertical"` entre
  `0.6` e `0.68` (abaixo do centro, ainda acima da área da legenda/botões do Instagram).

### 5. Renderizar e entregar
```bash
python scripts/legendar.py VIDEO.mp4 trabalho/frases_sinc.json -o saida/VIDEO_legendado.mp4 --largura 1080
```
`--largura 1080` entrega em 1080×1920, o tamanho máximo que o Instagram usa: o arquivo fica
bem menor (ex.: 564 MB em 1440p → 118 MB) sem diferença visível no Reel. Omita para manter a
resolução original (ex.: para arquivo/edição posterior).
Saem três arquivos lado a lado: o `.mp4` legendado (qualidade alta, CRF 18, áudio
preservado), o `.srt` (para CapCut/Premiere ou para subir no Instagram) e o `.ass`
(legenda estilizada, editável no Aegisub). Se algum aviso disser que uma frase foi reduzida
para caber, considere dividi-la e renderizar de novo. Entregue o vídeo e o `.srt` ao usuário.

## Fonte

A Helvetica Neue Condensed Bold do design é paga e não vem junto. A skill usa uma versão da
**Roboto Condensed Bold** (licença livre OFL, em `assets/fonts/`) estreitada para 91%, o que
deixa a mesma altura e largura da referência. Se o usuário tiver a Helvetica:

```bash
# Mac (vem no sistema):
python scripts/legendar.py VIDEO.mp4 frases.json --fonte /System/Library/Fonts/HelveticaNeue.ttc --fonte-nome "Condensed Bold"
# Arquivo avulso:
python scripts/legendar.py VIDEO.mp4 frases.json --fonte ~/Fonts/HelveticaNeue-CondensedBold.otf
```
Ao trocar a fonte, rode o `segmentar.py` com o mesmo estilo/fonte de antes da revisão se
quiser que as quebras levem em conta a nova largura (na prática a diferença é pequena e o
`legendar.py` encolhe sozinho qualquer frase que não caiba).

## Ajustes do estilo (`assets/estilo.json`)
| campo | o que faz |
|---|---|
| `cor` | cor do texto (#FFEA00) |
| `tamanho_px` | tamanho da letra num vídeo de 1080 de largura (70; o Figma tinha 86,7 e o usuário pediu menor) |
| `espacamento_letras_px` | espaço entre letras (−2,6 do Figma) |
| `escala_horizontal` | estreita a fonte reserva para imitar a Helvetica Condensed |
| `brilho_desfoque_px` | força do brilho (0 desliga) |
| `posicao_vertical` | 0.5 = centro; maior = mais para baixo |
| `largura_maxima` | fração da largura que uma frase pode ocupar (0,43 ≈ 2–3 palavras por vez; o usuário achou 0,84 texto demais por linha) |
| `sombra_contraste` | 0 = desligada; ~12 = sombra escura suave para fundos claros |

Tudo é proporcional: vídeos 720p ou 4K ficam com a mesma aparência de um 1080×1920.
