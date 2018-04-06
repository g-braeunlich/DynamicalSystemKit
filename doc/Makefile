file = getting_started.md
base = $(basename $(file))
pdf = $(addsuffix .pdf,$(base))
docx = $(addsuffix .docx,$(base))
figures = $(shell grep -oz '!\[[^]]*\]([^)]*' $(file) | grep -zPo '!\[[^]]*\]\(\K[^)]*' | tr '\0' '\n')
pdf_figures = $(addsuffix .pdf,$(figures))
emf_figures = $(addsuffix .emf,$(figures))

ifeq ($(shell pandoc --version | head -n1 | grep -Po ' \K[0-9]'),2)
  PANDOC = pandoc --number-sections -f "markdown+pipe_tables+smart+definition_lists" -F pandoc-crossref -F pandoc-citeproc
else
  PANDOC = pandoc --number-sections -S -f "markdown+pipe_tables+definition_lists" -F pandoc-crossref -F pandoc-citeproc
endif

all: $(pdf)

.PHONY: all

%.d: %.md
	awk '/```include/{getline ; getline file; print "$(<:.md=.md.included)" ": " $$file }' $< > $@

-include $(file:.md=.d)

$(pdf): $(pdf_figures) $(ext_pdf_figures)
$(docx): $(emf_figures) $(ext_emf_figures)

%.docx: %.md
	$(PANDOC) --default-image-extension=.emf -s $< -o $@

%.pdf: %.md
	$(PANDOC) --default-image-extension=.pdf -s $< -o $@

%.tex: %.md
	$(PANDOC) --default-image-extension=.pdf -s $< -o $@

%.pdf: %.svg
	inkscape -z $^ --export-pdf=$@

%.emf: %.svg
	inkscape -z $^ --export-emf=$@

