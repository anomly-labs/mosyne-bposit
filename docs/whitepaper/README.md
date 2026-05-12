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
- All `\cite` resolve to a `\bibitem` (and every `\bibitem` is cited)
- Compiles at home with `pdflatex` if available; otherwise relies on
  Overleaf's compile environment which has the standard LaTeX packages

## arXiv submission

`mosyne_bposit_whitepaper_arxiv.zip` is the single-file submission
bundle (just `mosyne_bposit_whitepaper.tex`, no README). arXiv's
TeX Live auto-compiles it; all packages used (`tikz`, `pgfplots`,
`hyperref`, `cleveref`, `booktabs`, `microtype`) are standard.

### Categories

- **Primary:** `cs.AR` (Hardware Architecture)
- **Cross-list:** `cs.MS` (Mathematical Software), `cs.LG` (Machine
  Learning — for the W8A8 Qwen result)
- Optional further cross-list: `cs.NA` (Numerical Analysis — for
  the determinism / quire arithmetic angle)

### Endorsement

First-time submitters in `cs.AR` typically need an endorsement from
someone with recent posting history in the category. Natural asks
given the bibliography:

- **John L. Gustafson** — directly cited; this is in his arena.
- **Stanford MINOTAUR group** (Prabhu, Raina, Mitra, et al.) — also
  cited; active in `cs.AR`.

### Metadata (copy-paste ready)

- **Title:** *Bounded Posits at IEEE Tensor-Core Throughput on Commodity NVIDIA Blackwell, with Bit-Exact Reproducibility*
- **Authors:** `Ry Bruscoe (Anomly Labs)`
- **Comments:** *10 pages, 4 figures, 5 tables. Reproducibility scripts and source at https://github.com/anomly-labs/mosyne-bposit*
- **License:** `CC BY 4.0` recommended (matches the spirit of the
  Apache-2.0 public source release); arXiv non-exclusive license is
  the default fallback.
- **MSC / ACM:** optional. Candidate ACM CCS:
  `Computer systems organization → Architectures → Single instruction, multiple data`.

### Pre-flight checklist

- [x] Single-file `.tex`, no external `.bib`, no missing figures
- [x] All `\bibitem`s cited
- [x] Author block in `\author{}` includes affiliation + email
- [x] Reproducibility URL points at the public GitHub repo
- [ ] ORCID slot in titlepage filled in (currently a commented TODO)
- [ ] Endorsement secured before first arXiv upload
