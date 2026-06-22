MAIN = thesis

.PHONY: all watch clean cleanall

all:
	latexmk $(MAIN).tex

watch:
	latexmk -pvc -view=none $(MAIN).tex

clean:
	latexmk -c

cleanall:
	latexmk -C
