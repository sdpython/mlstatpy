# -*- coding: utf-8 -*-
import sys
import os
from pyquickhelper.helpgen.default_conf import set_sphinx_variables, get_default_stylesheet

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

local_template = os.path.join(os.path.abspath(
    os.path.dirname(__file__)), "phdoc_templates")

set_sphinx_variables(__file__, "mlstatpy", "Xavier Dupré", 2021,
                     html_theme, html_theme_path, locals(),
                     extlinks=dict(
                         issue=('https://github.com/sdpython/mlstatpy/issues/%s', 'issue')),
                     title="Machine Learning, Statistiques et Programmation", book=True, nblayout='table')

# next

blog_root = "http://www.xavierdupre.fr/app/mlstatpy/helpsphinx/"

html_context = {
    'css_files': get_default_stylesheet() + ['_static/my-styles.css'],
}

html_logo = "phdoc_static/project_ico_small.png"

if choice == "bootstrap":
    html_theme_options = {
        'navbar_title': "BASE",
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
\\newcommand{\\scal}[2]{\\left<#1,#2\\right>}
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
# \\usepackage{eepic}

imgmath_latex_preamble = preamble + custom_preamble
latex_elements['preamble'] = preamble + custom_preamble
mathdef_link_only = True

epkg_dictionary.update({
    'ACP': 'https://fr.wikipedia.org/wiki/Analyse_en_composantes_principales',
    "AESA": "https://tavianator.com/aesa/",
    'ApproximateNMFPredictor':
        'http://www.xavierdupre.fr/app/mlinsights/helpsphinx/mlinsights/mlmodel/anmf_predictor.html',
    "B+ tree": "https://en.wikipedia.org/wiki/B%2B_tree",
    "Branch and Bound": "https://en.wikipedia.org/wiki/Branch_and_bound",
    "Custom Criterion for DecisionTreeRegressor":
        "http://www.xavierdupre.fr/app/mlinsights/helpsphinx/notebooks/piecewise_linear_regression_criterion.html",
    'cython': 'https://cython.org/',
    'DecisionTreeClassifier':
        'https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html',
    'DecisionTreeRegressor optimized for Linear Regression':
        "http://www.xavierdupre.fr/app/mlinsights/helpsphinx/notebooks/piecewise_linear_regression_criterion.html",
    'dot': 'https://fr.wikipedia.org/wiki/DOT_(langage)',
    'ICML 2016': 'https://icml.cc/2016/index.html',
    'KMeans': 'https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html',
    "LAESA": "https://tavianator.com/aesa/",
    'LAPACK': 'http://www.netlib.org/lapack/',
    'mlinsights': 'http://www.xavierdupre.fr/app/mlinsights/helpsphinx/index.html',
    'PiecewiseTreeRegressor':
        'http://www.xavierdupre.fr/app/mlinsights/helpsphinx/mlinsights/mlmodel/'
        'piecewise_tree_regression.html#mlinsights.mlmodel.piecewise_tree_regression.PiecewiseTreeRegressor',
    'Predictable t-SNE': 'http://www.xavierdupre.fr/app/mlinsights/helpsphinx/notebooks/predictable_tsne.html',
    "R-tree": "https://en.wikipedia.org/wiki/R-tree",
    "R* tree": "https://en.wikipedia.org/wiki/R*_tree",
    'Regression with confidence interval':
        'http://www.xavierdupre.fr/app/mlinsights/helpsphinx/notebooks/regression_confidence_interval.html',
    'ROC': 'https://fr.wikipedia.org/wiki/Courbe_ROC',
    'statsmodels': 'http://www.statsmodels.org/stable/index.html',
    'SVD': 'https://fr.wikipedia.org/wiki/D%C3%A9composition_en_valeurs_singuli%C3%A8res',
    'Visualize a scikit-learn pipeline':
        'http://www.xavierdupre.fr/app/mlinsights/helpsphinx/notebooks/visualize_pipeline.html',
    "X-tree": "https://en.wikipedia.org/wiki/X-tree",
})

nblinks = {
    'l-reglin-piecewise-streaming':
        'http://www.xavierdupre.fr/app/mlstatpy/helpsphinx/c_ml/piecewise.html#streaming-linear-regression',
}
