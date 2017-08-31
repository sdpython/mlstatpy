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


class TestRst2Html(unittest.TestCase):

    def test_rst_syntax(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        this = os.path.abspath(os.path.dirname(__file__))
        rst = os.path.join(this, "..", "..", "_doc", "sphinxdoc",
                           "source", "c_garden", "file_dattente.rst")
        if not os.path.exists(rst):
            raise FileNotFoundError(rst)
        with open(rst, "r", encoding="utf-8") as f:
            content = f.read()
        content = content.replace(".. toctree::", "")
        content = content.replace(".. contents::", "")
        content = content.replace("    :local:", "")
        try:
            ht = rst2html(content, writer="rst", layout="sphinx",
                          keep_warnings=True)
            fLOG(ht)
        except Exception as e:
            fLOG(e)


if __name__ == "__main__":
    unittest.main()
