# Preprint source

`main.tex` + `refs.bib` + `figs/`. Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main`.
Figures: `cd src && python analysis_paper.py --run gold && python analysis_paper.py --run product`.
Numbers: `outputs/gold/paper_stats.json`, `outputs/product/paper_stats.json`, and each run's `summary.json`.
