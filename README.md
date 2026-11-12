# Legendas NM

Skill do Claude para legendar Reels da igreja / ministério de louvor no estilo da casa:
uma frase por vez, centralizada, amarelo #FFEA00, Helvetica Neue Condensed Bold com brilho.

A skill está em [`legendas-reels/`](legendas-reels/SKILL.md).

## Como instalar a skill
- **Claude (app/site):** baixe o arquivo `legendas-reels.skill` (pasta `dist/`) e, em
  Configurações → Capacidades → Skills, faça o upload. Ou abra o arquivo numa conversa e clique em "Salvar skill".
- **Claude Code:** copie a pasta `legendas-reels` para `~/.claude/skills/`.

## Requisitos (onde o vídeo for processado)
```bash
pip install -r legendas-reels/requirements.txt
```
e o `ffmpeg` instalado (`brew install ffmpeg` no Mac).

## Como usar
Mande o vídeo para o Claude e peça: "legenda esse reels". Ele transcreve, revisa os nomes
bíblicos, mostra as frases para você conferir e devolve o vídeo legendado + o arquivo `.srt`.
