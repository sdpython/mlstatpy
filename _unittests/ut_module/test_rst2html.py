#-*- coding: utf-8 -*-
"""
@brief      test log(time=38s)
"""

import sys
import os
import unittest


try:
    import pyquickhelper as skip_
except ImportError:
    path = os.path.normpath(
        os.path.abspath(
            os.path.join(
                os.path.split(__file__)[0],
                "..",
                "..",
                "..",
                "pyquickhelper",
                "src")))
    if path not in sys.path:
        sys.path.append(path)
    import pyquickhelper as skip_


try:
    import src
except ImportError:
    path = os.path.normpath(
        os.path.abspath(
            os.path.join(
                os.path.split(__file__)[0],
                "..",
                "..")))
    if path not in sys.path:
        sys.path.append(path)
    import src

from pyquickhelper.loghelper import fLOG
from pyquickhelper.helpgen import rst2html
from pyquickhelper.pycode import get_temp_folder, is_travis_or_appveyor


class TestRst2Html(unittest.TestCase):

    preamble = '''
            \\usepackage{etex}
            \\usepackage{fixltx2e} % LaTeX patches, \\textsubscript
            \\usepackage{cmap} % fix search and cut-and-paste in Acrobat
            \\usepackage[raccourcis]{fast-diagram}
            \\usepackage{titlesec}
            \\usepackage{amsmath}
            \\usepackage{amssymb}
            \\usepackage{amsfonts}
            \\usepackage{graphics}
            \\usepackage{epic}
            \\usepackage{eepic}
            %\\usepackage{pict2e}
            %%% Redefined titleformat
            \\setlength{\\parindent}{0cm}
            \\setlength{\\parskip}{1ex plus 0.5ex minus 0.2ex}
            \\newcommand{\\hsp}{\\hspace{20pt}}
            \\newcommand{\\acc}[1]{\\left\\{#1\\right\\}}
            \\newcommand{\\cro}[1]{\\left[#1\\right]}
            \\newcommand{\\pa}[1]{\\left(#1\\right)}
            \\newcommand{\\R}{\\mathbb{R}}
            \\newcommand{\\HRule}{\\rule{\\linewidth}{0.5mm}}
            %\\titleformat{\\chapter}[hang]{\\Huge\\bfseries\\sffamily}{\\thechapter\\hsp}{0pt}{\\Huge\\bfseries\\sffamily}
            '''.replace("            ", "")

    custom_preamble = """\n
            \\usepackage[all]{xy}
            \\newcommand{\\vecteur}[2]{\\pa{#1,\\dots,#2}}
            \\newcommand{\\N}[0]{\\mathbb{N}}
            \\newcommand{\\indicatrice}[1]{\\mathbf{1\\!\\!1}_{\\acc{#1}}}
            \\newcommand{\\infegal}[0]{\\leqslant}
            \\newcommand{\\supegal}[0]{\\geqslant}
            \\newcommand{\\ensemble}[2]{\\acc{#1,\\dots,#2}}
            \\newcommand{\\fleche}[1]{\\overrightarrow{ #1 }}
            \\newcommand{\\intervalle}[2]{\\left\\{#1,\\cdots,#2\\right\\}}
            \\newcommand{\\independant}[0]
            {\\;\\makebox[3ex]{\\makebox[0ex]{\\rule[-0.2ex]{3ex}{.1ex}}\\!\\!\\!\\!\\makebox[.5ex][l]
            {\\rule[-.2ex]{.1ex}{2ex}}\\makebox[.5ex][l]{\\rule[-.2ex]{.1ex}{2ex}}} \\,\\,}
            \\newcommand{\\esp}{\\mathbb{E}}
            \\newcommand{\\espf}[2]{\\mathbb{E}_{#1}\\pa{#2}}
            \\newcommand{\\var}{\\mathbb{V}}
            \\newcommand{\\pr}[1]{\\mathbb{P}\\pa{#1}}
            \\newcommand{\\loi}[0]{{\\cal L}}
            \\newcommand{\\vecteurno}[2]{#1,\\dots,#2}
            \\newcommand{\\norm}[1]{\\left\\Vert#1\\right\\Vert}
            \\newcommand{\\norme}[1]{\\left\\Vert#1\\right\\Vert}
            \\newcommand{\\dans}[0]{\\rightarrow}
            \\newcommand{\\partialfrac}[2]{\\frac{\\partial #1}{\\partial #2}}
            \\newcommand{\\partialdfrac}[2]{\\dfrac{\\partial #1}{\\partial #2}}
            \\newcommand{\\trace}[1]{tr\\pa{#1}}
            \\newcommand{\\sac}[0]{|}
            \\newcommand{\\abs}[1]{\\left|#1\\right|}
            \\newcommand{\\loinormale}[2]{{\\cal N} \\pa{#1,#2}}
            \\newcommand{\\loibinomialea}[1]{{\\cal B} \\pa{#1}}
            \\newcommand{\\loibinomiale}[2]{{\\cal B} \\pa{#1,#2}}
            \\newcommand{\\loimultinomiale}[1]{{\\cal M} \\pa{#1}}
            \\newcommand{\\variance}[1]{\\mathbb{V}\\pa{#1}}
            """.replace("            ", "")

    def test_rst_syntax(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        temp = get_temp_folder(__file__, "temp_rst_syntax")
        preamble = TestRst2Html.preamble + TestRst2Html.custom_preamble
        this = os.path.abspath(os.path.dirname(__file__))
        rst = os.path.join(this, "..", "..", "_doc", "sphinxdoc",
                           "source", "c_garden", "strategie_avec_alea.rst")
        if not os.path.exists(rst):
            raise FileNotFoundError(rst)
        with open(rst, "r", encoding="utf-8") as f:
            content = f.read()
        writer = "html"

        if is_travis_or_appveyor() in ('travis', 'appveyor'):
            # No latex.
            return

        sys.path.append(os.path.abspath(os.path.dirname(src.__file__)))

        ht = rst2html(content, writer=writer, layout="sphinx",
                      keep_warnings=True, imgmath_latex_preamble=preamble,
                      outdir=temp)
        sys.path.pop()

        rst = os.path.join(temp, "out.{0}".format(writer))
        with open(rst, "w", encoding="utf-8") as f:
            f.write(ht)


if __name__ == "__main__":
    unittest.main()
