import sys
import os
from sphinx_runpython.github_link import make_linkcode_resolve
from sphinx_runpython.conf_helper import has_dvipng, has_dvisvgm
from mlstatpy import __version__


extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx_gallery.gen_gallery",
    "sphinx_issues",
    "sphinx_runpython.blocdefs.sphinx_exref_extension",
    "sphinx_runpython.blocdefs.sphinx_faqref_extension",
    "sphinx_runpython.blocdefs.sphinx_mathdef_extension",
    "sphinx_runpython.epkg",
    "sphinx_runpython.gdot",
    "sphinx_runpython.runpython",
    "matplotlib.sphinxext.plot_directive",
]

if has_dvisvgm():
    extensions.append("sphinx.ext.imgmath")
    imgmath_image_format = "svg"
elif has_dvipng():
    extensions.append("sphinx.ext.pngmath")
    imgmath_image_format = "png"
else:
    extensions.append("sphinx.ext.mathjax")

templates_path = ["_templates"]
html_logo = "_static/project_ico.png"
source_suffix = ".rst"
master_doc = "index"
project = "mlstatpy"
copyright = "2016-2025, Xavier Dupré"
author = "Xavier Dupré"
version = __version__
release = __version__
language = "fr"
exclude_patterns = ["auto_examples/*.ipynb"]
pygments_style = "sphinx"
todo_include_todos = True
nbsphinx_execute = "never"

html_theme = "furo"
html_theme_path = ["_static"]
html_theme_options = {}
html_sourcelink_suffix = ""
html_static_path = ["_static"]

issues_github_path = "sdpython/mlstatpy"

nbsphinx_prolog = """

.. _nbl-{{ env.doc2path(env.docname, base=None).replace("/", "-").split(".")[0] }}:

"""

nbsphinx_epilog = """
----

`Notebook on github <https://github.com/sdpython/teachpyx/tree/main/_doc/{{ env.doc2path(env.docname, base=None) }}>`_
"""

# The following is used by sphinx.ext.linkcode to provide links to github
linkcode_resolve = make_linkcode_resolve(
    "mlstatpy",
    (
        "https://github.com/sdpython/mlstatpy/"
        "blob/{revision}/{package}/"
        "{path}#L{lineno}"
    ),
)

latex_elements = {
    "papersize": "a4",
    "pointsize": "10pt",
    "title": project,
}

mathjax3_config = {"chtml": {"displayAlign": "left"}}

intersphinx_mapping = {
    "onnx": ("https://onnx.ai/onnx/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "python": (f"https://docs.python.org/{sys.version_info.major}", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "sklearn-onnx": ("https://onnx.ai/sklearn-onnx/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# Check intersphinx reference targets exist
nitpicky = True
# See also scikit-learn/scikit-learn#26761
nitpick_ignore = [
    ("py:class", "False"),
    ("py:class", "True"),
    ("py:class", "pipeline.Pipeline"),
    ("py:class", "default=sklearn.utils.metadata_routing.UNCHANGED"),
]

sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": os.path.join(os.path.dirname(__file__), "examples"),
    # path where to save gallery generated examples
    "gallery_dirs": "auto_examples",
}

# next

preamble = """
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

\\usepackage[all]{xy}
\\newcommand{\\vecteur}[2]{\\pa{#1,\\dots,#2}}
\\newcommand{\\N}[0]{\\mathbb{N}}
\\newcommand{\\indicatrice}[1]{ {1\\!\\!1}_{\\acc{#1}} }
\\newcommand{\\infegal}[0]{\\leqslant}
\\newcommand{\\supegal}[0]{\\geqslant}
\\newcommand{\\ensemble}[2]{\\acc{#1,\\dots,#2}}
\\newcommand{\\fleche}[1]{\\overrightarrow{ #1 }}
\\newcommand{\\intervalle}[2]{\\left\\{#1,\\cdots,#2\\right\\}}
\\newcommand{\\independant}[0]{\\perp \\!\\!\\! \\perp}
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
\\newcommand{\\intf}[1]{\\left\\lfloor #1 \\right\\rfloor}
"""

epkg_dictionary = {
    "ACP": "https://fr.wikipedia.org/wiki/Analyse_en_composantes_principales",
    "AESA": "https://tavianator.com/aesa/",
    "ApproximateNMFPredictor": "https://sdpython.github.io/doc/mlinsights/dev/api/mlmodel.html",
    "AUC": "https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve",
    "B+ tree": "https://en.wikipedia.org/wiki/B%2B_tree",
    "BLAS": "https://www.netlib.org/blas/",
    "Branch and Bound": "https://en.wikipedia.org/wiki/Branch_and_bound",
    "C++": "https://fr.wikipedia.org/wiki/C%2B%2B",
    "Custom Criterion for DecisionTreeRegressor": "https://sdpython.github.io/doc/mlinsights/dev/auto_examples/plot_piecewise_linear_regression_criterion.html",
    "cython": "https://cython.org/",
    "DecisionTreeClassifier": "https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html",
    "DecisionTreeRegressor optimized for Linear Regression": "https://sdpython.github.io/doc/mlinsights/dev/auto_examples/plot_piecewise_linear_regression_criterion.html",
    "dot": "https://fr.wikipedia.org/wiki/DOT_(langage)",
    "Holm-Bonferroni method": "https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method",
    "ICML 2016": "https://icml.cc/2016/index.html",
    "KMeans": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html",
    "LAESA": "https://tavianator.com/aesa/",
    "LAPACK": "http://www.netlib.org/lapack/",
    "mlinsights": "https://sdpython.github.io/doc/mlinsights/dev/index.html",
    "mlstatpy": "https://sdpython.github.io/doc/mlstatpy/dev/",
    "numpy": (
        "https://www.numpy.org/",
        ("https://docs.scipy.org/doc/numpy/reference/generated/numpy.{0}.html", 1),
        ("https://docs.scipy.org/doc/numpy/reference/generated/numpy.{0}.{1}.html", 2),
    ),
    "PiecewiseTreeRegressor": "https://sdpython.github.io/doc/mlinsights/dev/api/mlmodel_tree.html#piecewisetreeregressor",
    "Pillow": "https://pillow.readthedocs.io/en/stable/",
    "Predictable t-SNE": "https://sdpython.github.io/doc/mlinsights/dev/auto_examples/plot_predictable_tsne.html",
    "QuantileLinearRegression": "https://sdpython.github.io/doc/mlinsights/dev/api/mlmodel.html#quantilelinearregression",
    "R-tree": "https://en.wikipedia.org/wiki/R-tree",
    "R* tree": "https://en.wikipedia.org/wiki/R*_tree",
    "Regression with confidence interval": "https://sdpython.github.io/doc/mlinsights/dev/auto_examples/plot_regression_confidence_interval.html",
    "relu": "https://en.wikipedia.org/wiki/Rectifier_(neural_networks)",
    "ROC": "https://fr.wikipedia.org/wiki/Courbe_ROC",
    "scikit-learn": "https://scikit-learn.org/stable/index.html",
    "sklearn": "https://scikit-learn.org/stable/index.html",
    "sklearn-onnx": "https://onnx.ai/sklearn-onnx/",
    "statsmodels": "http://www.statsmodels.org/stable/index.html",
    "SVD": "https://fr.wikipedia.org/wiki/D%C3%A9composition_en_valeurs_singuli%C3%A8res",
    "tqdm": "https://tqdm.github.io/",
    "Visualize a scikit-learn pipeline": "https://sdpython.github.io/doc/mlinsights/dev/auto_examples/plot_visualize_pipeline.html",
    "X-tree": "https://en.wikipedia.org/wiki/X-tree",
    "wikipedia dumps": "https://dumps.wikimedia.org/frwiki/latest/",
}

imgmath_latex_preamble = preamble
latex_elements["preamble"] = imgmath_latex_preamble
