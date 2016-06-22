#-*- coding: utf-8 -*-
import sys
import os
import datetime
import re
import sphinx_theme_pd as sphtheme
# import sphinx_clatex
# import hbp_sphinx_theme as sphtheme

sys.path.insert(0, os.path.abspath(os.path.join(os.path.split(__file__)[0])))
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.split(__file__)[0],
            "..",
            "..",
            "..",
            "..",
            "pyquickhelper",
            "src")))

from pyquickhelper.helpgen.default_conf import set_sphinx_variables
set_sphinx_variables(__file__, "mlstatpy", "Xavier Dupré", 2016,
                     sphtheme.__name__, [sphtheme.get_html_theme_path()], locals(),
                     extlinks=dict(issue=('https://github.com/sdpython/mlstatpy/issues/%s', 'issue')),
                     title="Machine Learning, Statistiques et Programmation", book=True)

blog_root = "http://www.xavierdupre.fr/app/mlstatpy/helpsphinx/"

html_context = {
    'css_files': ['_static/my-styles.css'],
}

language = "fr"
custom_preamble = """\n\\newcommand{\\girafedec}[3]{ \\begin{array}{ccccc} #1 &=& #2 &+& #3 \\\\ a' &=& a &-& o  \\end{array}}
            \\newcommand{\\vecteur}[2]{\\pa{#1,\\dots,#2}}
            \\newcommand{\\N}[0]{\\mathbb{N}}
            \\newcommand{\\indicatrice}[1]{\\mathbf{1\\!\\!1}_{\\acc{#1}}}
            \\usepackage[all]{xy}
            \\newcommand{\\infegal}[0]{\\leqslant}
            \\newcommand{\\supegal}[0]{\\geqslant}
            \\newcommand{\\ensemble}[2]{\\acc{#1,\\dots,#2}}
            \\newcommand{\\fleche}[1]{\\overrightarrow{ #1 }}
            \\newcommand{\\intervalle}[2]{\\left\\{#1,\\cdots,#2\\right\\}}
            """
imgmath_latex_preamble += custom_preamble
preamble += custom_preamble

