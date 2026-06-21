# Use LuaLaTeX: the thesis mixes Greek and Latin via fontspec + TeX Gyre
# Termes, which needs a Unicode engine.  (1 = pdflatex, 4 = lualatex, 5 = xelatex.)
$pdf_mode = 4;

# Make the engine emit SyncTeX data (thesis.synctex.gz) so Emacs/pdf-tools
# forward search (C-c C-v) and reverse search (Ctrl/double-click) work --
# latexmk does not enable synctex by default.
$synctex = 1;
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
$lualatex = 'lualatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
$xelatex  = 'xelatex  -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
