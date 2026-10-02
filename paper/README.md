# The Step 9 report

`main.tex` is the report skeleton: the structure, figures, tables and numbers are generated, and
the text is written by hand. Each section lists in comments what it has to contain.

## Regenerate figures, tables and numbers

```powershell
python s2_classify.py          # CV results (the blind test result is kept from the one --test run)
python s2_paper.py             # paper/figures/*.pdf and paper/generated/*.tex
```

No result is typed by hand: the text uses macros from `generated/numbers.tex`, such as `\TestAcc`
(89\%) or `\NTrain` (68), so a re-run updates every number in the report at once.

## Compile on Overleaf

1. Zip this `paper` folder (right-click → Send to → Compressed folder).
2. On overleaf.com: **New Project → Upload Project**, choose the zip.
3. Compiler: pdfLaTeX (the default). Overleaf runs BibTeX for the references by itself.

Red `[TODO: …]` marks show what is still to be written.

## References

Every entry in `references.bib` was checked against its DOI record. Add a reference only after
finding and checking it yourself; never cite from memory.
