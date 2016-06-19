#-*- coding: utf-8 -*-
import sys
import os
import datetime
import re
import sphinx_theme_pd as sphtheme
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
