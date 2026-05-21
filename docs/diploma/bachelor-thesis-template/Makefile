MAIN := main

.PHONY: pdf clean distclean check-env

pdf: check-env
	latexmk -pdf $(MAIN).tex

clean:
	latexmk -c $(MAIN).tex

distclean:
	latexmk -C $(MAIN).tex

check-env:
	@command -v latexmk >/dev/null || { echo "latexmk is not installed"; exit 1; }
	@command -v pdflatex >/dev/null || { echo "pdflatex is not installed"; exit 1; }
	@command -v biber >/dev/null || { echo "biber is not installed"; exit 1; }
