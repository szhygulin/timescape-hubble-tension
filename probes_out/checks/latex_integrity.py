import re, collections
tex = open("timescape-hubble-tension.tex").read()
lines = tex.split("\n")

# 1. cite keys vs bibitems
cite_keys = set()
for m in re.finditer(r'\\cite\{([^}]*)\}', tex):
    for k in m.group(1).split(','):
        cite_keys.add(k.strip())
bib_keys = set(re.findall(r'\\bibitem\{([^}]*)\}', tex))
print("CITE keys:", len(cite_keys))
print("BIB keys:", len(bib_keys))
print("cited-but-no-bibitem:", sorted(cite_keys - bib_keys))
print("bibitem-but-uncited:", sorted(bib_keys - cite_keys))

# 2. begin/end environments
begins = collections.Counter(re.findall(r'\\begin\{([^}]*)\}', tex))
ends = collections.Counter(re.findall(r'\\end\{([^}]*)\}', tex))
print("\nENV imbalance:")
for e in set(begins)|set(ends):
    if begins[e]!=ends[e]:
        print("  ", e, begins[e], ends[e])
print("  (none above = balanced)")

# 3. dollar parity (strip escaped \$)
t2 = tex.replace(r'\$','')
nd = t2.count('$')
print("\nInline $ count (excl escaped):", nd, "parity", "OK-even" if nd%2==0 else "ODD-BAD")

# 4. braces (naive, strip escaped)
t3 = re.sub(r'\\[{}]','', tex)
print("Brace balance { } :", t3.count('{'), t3.count('}'), "OK" if t3.count('{')==t3.count('}') else "MISMATCH")

# 5. document/abstract singletons
print("\n\\begin{document}:", tex.count(r'\begin{document}'), " \\end{document}:", tex.count(r'\end{document}'))
print("abstract env:", tex.count(r'\begin{abstract}'), tex.count(r'\end{abstract}'))

# 6. footnotes location: are any inside abstract?
abs_start = tex.find(r'\begin{abstract}'); abs_end = tex.find(r'\end{abstract}')
abstract = tex[abs_start:abs_end]
print("\\footnote in abstract:", abstract.count(r'\footnote'))
print("total \\footnote:", tex.count(r'\footnote'))

# 7. tabular column consistency
for m in re.finditer(r'\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}', tex, re.S):
    spec = m.group(1)
    ncol = sum(1 for c in spec if c in 'lcr')
    body = m.group(2)
    # count & per data row (rows with \\)
    rows = [r for r in body.split(r'\\') ]
    print(f"\ntabular spec '{spec}' -> {ncol} cols")
    for r in rows:
        rr = r.strip()
        if not rr or rr.startswith('%'): continue
        if r'\colrule' in rr or r'\hline' in rr:
            rr = rr.split(r'\colrule')[-1].split(r'\hline')[-1].strip()
        if not rr: continue
        # skip pure macro lines
        amp = rr.count('&') - rr.count(r'\&')
        if amp>0:
            firsttok = rr[:30].replace("\n"," ")
            if amp+1 != ncol:
                print(f"   ROW COLS={amp+1} (!=%d) :: {firsttok!r}"%ncol)
