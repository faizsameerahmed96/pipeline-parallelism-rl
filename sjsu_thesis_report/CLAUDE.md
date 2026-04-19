# CLAUDE.md

## Overview

SJSU thesis report using the [SJSU-thesis-LaTeX](https://github.com/taustin/SJSU-thesis-LaTeX) template. Content comes from experiments in `../experiments/exp2-act-grad-acc/`. Figures and data are generated from experiment results and placed into this report.

## Hosted on Overleaf

This project is synced with Overleaf. Keep it compatible:
- No local-only packages or custom build scripts
- Use only standard LaTeX packages available on Overleaf
- All figures go in `images/` as PDF/PNG (no symlinks)
- BibTeX references in `references.bib`

## Build Commands

```bash
pdflatex thesis
bibtex thesis
pdflatex thesis
pdflatex thesis
```

## Structure

- `thesis.tex` — Main document entry point
- `abs.tex` — Abstract
- `ack.tex` — Acknowledgements
- `chap1.tex` through `chap5.tex` — Chapters (Intro, Related Work, Problem, Method, Evaluation)
- `references.bib` — Bibliography
- `images/` — Figures (copied from `../report/figures/`)
- `urithesis.cls`, `uriref.bst`, `uribib.bst` — Template class/style files (do not modify)
