# IGARSS 2027 paper

The 4-page IEEE conference paper for IGARSS 2027 (portal opens 10 November 2026, deadline
11 January 2027). The text in `main.tex` is written by hand; everything else is generated.

```powershell
python s2_igarss.py      # figures/ and generated/ (numbers, tables), at IEEE column width
```

- `main.tex`: IEEEtran conference skeleton. Each section lists in comments what it must contain,
  its page budget and the macros for its numbers. Red `[TODO]` marks are text still to write;
  red `[pending]` marks are field-visit results.
- After the field visit: save the form as `labels/ground_truth.csv`, run `python s2_ground.py`, then
  `python s2_igarss.py`. The farmer row of Table I, the `\Ground...` numbers and a third panel in the
  confusion figure appear by themselves.
- `references.bib`: 13 references, each checked against its DOI or arXiv record.
- Compile on Overleaf (upload the folder as a zip; pdfLaTeX) or locally with Tectonic:
  `tectonic -X compile main.tex`.
- Before submitting: replace the official 2027 paper kit's template if it differs, delete `\nocite{*}`
  once the text cites the references, and write the AI-use statement in the acknowledgment.
