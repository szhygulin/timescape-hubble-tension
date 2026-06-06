#!/bin/bash
# Install a no-sudo TeX stack (TinyTeX) sufficient to compile the revtex4-2
# manuscript, then add the packages it needs. Idempotent-ish.
set +e
echo "=== checking for an existing TeX ==="
EXISTING=""
for d in /Library/TeX/texbin /usr/local/texlive/*/bin/* "$HOME/Library/TinyTeX/bin/"* "$HOME/.TinyTeX/bin/"*; do
  [ -x "$d/pdflatex" ] && EXISTING="$d" && echo "found pdflatex at $d"
done

if [ -z "$EXISTING" ] && ! command -v pdflatex >/dev/null 2>&1; then
  echo "=== installing TinyTeX (user-space, no sudo) ==="
  curl -sL "https://yihui.org/tinytex/install-bin-unix.sh" | sh
fi

# locate the TeX bin dir
TLBIN=""
for d in "$HOME/Library/TinyTeX/bin/"* "$HOME/.TinyTeX/bin/"* /Library/TeX/texbin /usr/local/texlive/*/bin/*; do
  [ -x "$d/tlmgr" ] && TLBIN="$d" && break
done
export PATH="$TLBIN:/Library/TeX/texbin:$PATH"
echo "=== using TeX bin: $TLBIN ==="

echo "=== installing packages for revtex4-2 manuscript ==="
tlmgr update --self 2>&1 | tail -2
tlmgr install \
  revtex revtex4 latexbug \
  collection-latexrecommended collection-mathscience collection-fontsrecommended \
  amsmath amscls amsfonts mathtools \
  booktabs hyperref geometry xcolor graphics tools \
  url natbib bm textcase 2>&1 | tail -20

echo "=== pdflatex version ==="
pdflatex --version | head -2
# persist PATH hint for later compiles
echo "$TLBIN" > "$HOME/research/timescape-hubble-tension/src/.texbin"
echo "TEX_SETUP_DONE"
