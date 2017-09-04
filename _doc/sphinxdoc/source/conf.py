#-*- coding: utf-8 -*-
import sys
import os
import datetime
import re
# import sphinx_clatex
# import hbp_sphinx_theme as sphtheme

choice = "bootstrap"

if choice == "sphtheme":
    import sphinx_theme_pd as sphtheme
    html_theme = sphtheme.__name__
    html_theme_path = [sphtheme.get_html_theme_path()]
elif choice == "bootstrap":
    import sphinx_bootstrap_theme
    html_theme = 'bootstrap'
    html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()
else:
    raise NotImplementedError()

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

local_template = os.path.join(os.path.abspath(
    os.path.dirname(__file__)), "phdoc_templates")

from pyquickhelper.helpgen.default_conf import set_sphinx_variables, get_default_stylesheet
set_sphinx_variables(__file__, "mlstatpy", "Xavier Dupré", 2017,
                     html_theme, html_theme_path, locals(),
                     extlinks=dict(
                         issue=('https://github.com/sdpython/mlstatpy/issues/%s', 'issue')),
                     title="Machine Learning, Statistiques et Programmation", book=True)

# next

blog_root = "http://www.xavierdupre.fr/app/mlstatpy/helpsphinx/"

html_context = {
    'css_files': get_default_stylesheet() + ['_static/my-styles.css'],
}

html_logo = "project_ico_small.png"

html_sidebars = {}

if choice == "bootstrap":
    html_theme_options = {
        'navbar_title': ".",
        'navbar_site_name': "Site",
        'navbar_links': [
            ("XD", "http://www.xavierdupre.fr", True),
            ("blog", "blog/main_0000.html", True),
            ("index", "genindex"),
        ],
        'navbar_sidebarrel': True,
        'navbar_pagenav': True,
        'navbar_pagenav_name': "Page",
        'bootswatch_theme': "readable",
        # united = weird colors, sandstone=green, simplex=red, paper=trop bleu
        # lumen: OK
        # to try, yeti, flatly, paper
        'bootstrap_version': "3",
        'source_link_position': "footer",
    }

language = "fr"

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
'''

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
\\newcommand{\\independant}[0]{\\;\\makebox[3ex]{\\makebox[0ex]{\\rule[-0.2ex]{3ex}{.1ex}}\\!\\!\\!\\!\\makebox[.5ex][l]{\\rule[-.2ex]{.1ex}{2ex}}\\makebox[.5ex][l]{\\rule[-.2ex]{.1ex}{2ex}}} \\,\\,}
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
"""
#\\usepackage{eepic}

imgmath_latex_preamble = preamble + custom_preamble
latex_elements['preamble'] = preamble + custom_preamble
mathdef_link_only = True
