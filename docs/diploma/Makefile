THESIS_DIR := bachelor-thesis-template
MAIN := main

.PHONY: pdf clean distclean check-env

pdf: check-env
	$(MAKE) -C $(THESIS_DIR) pdf

clean:
	$(MAKE) -C $(THESIS_DIR) clean

distclean:
	$(MAKE) -C $(THESIS_DIR) distclean

check-env:
	@command -v latexmk >/dev/null || { echo "latexmk is not installed"; exit 1; }
	@command -v pdflatex >/dev/null || { echo "pdflatex is not installed"; exit 1; }
	@command -v biber >/dev/null || { echo "biber is not installed"; exit 1; }
