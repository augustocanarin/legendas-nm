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
pip install faster-whisper pillow fonttools
ffmpeg -version   # precisa existir; no Mac: brew install ffmpeg
```

Na primeira transcrição, o faster-whisper baixa o modelo de voz do site huggingface.co
(~1,5 GB no `medium`). Se o download for bloqueado pela rede, avise o usuário: é preciso
liberar o host `huggingface.co` (em sessões na nuvem, nas configurações do ambiente).

Os scripts ficam em `scripts/` desta skill (nos comandos abaixo, `scripts/` é relativo à pasta
da skill; use o caminho completo). Salve os arquivos de trabalho numa pasta própria do vídeo,
nunca dentro da skill.

## Fluxo

### 1. Entender o vídeo
Confirme o arquivo e pergunte só o que mudar o resultado e não for óbvio:
- Quem fala e quais nomes próprios aparecem (pregador, igreja, música, livro/versículo citado).
  Isso vira a `--dica` da transcrição e melhora muito a precisão.
- Se há trechos **cantados**: por padrão, legende as falas e também a letra cantada. O Whisper
  erra bastante em música com banda; se o usuário souber a música, peça a letra e use-a para
  corrigir o texto (mantendo os tempos).

### 2. Transcrever
```bash
python scripts/transcrever.py VIDEO.mp4 -o trabalho/transcricao.json --dica "Pr. Fulano, Filipenses 3, Getsêmani"
```
Modelo padrão `medium` (bom equilíbrio para português). Use `--modelo large-v3` se a
transcrição sair ruim e houver tempo; `small` se a máquina for fraca. O script lista as
palavras de baixa confiança — elas são o primeiro lugar a revisar.

### 3. Quebrar em frases e revisar o texto
```bash
python scripts/segmentar.py trabalho/transcricao.json -o trabalho/frases.json
```
O script já garante que cada frase cabe numa linha e evita cortes feios. Agora **leia
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

Mostre ao usuário a lista final de frases (número, tempo e texto) e as correções que você fez
e que mudam sentido ou nome — ele conhece o conteúdo e pega erros que você não pegaria. Em
pedidos do tipo "só legenda e me manda", siga em frente e liste as correções no final.

### 4. Conferir o visual antes de renderizar
```bash
python scripts/legendar.py VIDEO.mp4 trabalho/frases.json -o saida/VIDEO_legendado.mp4 --previa
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
python scripts/legendar.py VIDEO.mp4 trabalho/frases.json -o saida/VIDEO_legendado.mp4
```
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
| `tamanho_px` | tamanho da letra num vídeo de 1080 de largura (86,7 do Figma) |
| `espacamento_letras_px` | espaço entre letras (−2,6 do Figma) |
| `escala_horizontal` | estreita a fonte reserva para imitar a Helvetica Condensed |
| `brilho_desfoque_px` | força do brilho (0 desliga) |
| `posicao_vertical` | 0.5 = centro; maior = mais para baixo |
| `largura_maxima` | fração da largura do vídeo que uma frase pode ocupar |
| `sombra_contraste` | 0 = desligada; ~12 = sombra escura suave para fundos claros |

Tudo é proporcional: vídeos 720p ou 4K ficam com a mesma aparência de um 1080×1920.
