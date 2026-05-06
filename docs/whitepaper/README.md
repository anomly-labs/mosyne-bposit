# Mosyne · bposit-on-Blackwell whitepaper

`mosyne_bposit_whitepaper.tex` — single-file LaTeX whitepaper with all
results, ready to upload to Overleaf.

## Upload to Overleaf

1. Sign in at <https://www.overleaf.com>.
2. New Project → Upload Project → upload just `mosyne_bposit_whitepaper.tex`.
3. Compile (default pdflatex). Should produce a 6–8 page PDF.
4. Share with John Gustafson (`Share` → email).

The file is fully self-contained — no external `.bib`, no images, no
custom packages beyond standard `amsmath`, `booktabs`, `hyperref`,
`cleveref`, `listings`. Compiles on any default Overleaf LaTeX setup.

## What's in it

- Title + abstract + Gustafson endorsement
- Method: Path C pipeline + the 8-primitive bposit ALU
- Results: matmul shape sweep, determinism head-to-head, real Qwen
  layer, W8A8 PTQ progression, 5-DST physics suite
- Related work: MINOTAUR + Aspen + SmoothQuant/AWQ + NVIDIA non-det docs
- Honest limitations
- Acknowledgments + reproducibility section + 11-entry bibliography

## Verified before commit

- All environments (`\begin`/`\end`) balanced
- All `\ref` / `\Cref` resolve to a `\label`
- All `\cite` resolve to a `\bibitem`
- Compiles at home with `pdflatex` if available; otherwise relies on
  Overleaf's compile environment which has the standard LaTeX packages
