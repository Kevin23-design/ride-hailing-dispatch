#!/bin/bash
# 组装 main.tex：模板导言区 + 标题 + 摘要 + 各章节 + 参考文献，然后用 XeLaTeX 编译
cd "$(dirname "$0")"
{ cat preamble.tex body_head.tex; echo '\input{tables/exp24_macros.tex}'; echo '\input{tables/exp5_macros.tex}'; cat abstract.tex
  for f in 1.引言 2.相关工作 3.方法 4.实验 5.结论; do echo "\\input{texfiles/$f.tex}"; done
  cat refs.tex; echo '\end{document}'; } > main.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex > build.log 2>&1; echo "exit $?"
