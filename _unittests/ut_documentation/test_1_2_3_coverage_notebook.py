#-*- coding: utf-8 -*-
"""
@brief      test log(time=21s)
"""

import sys
import os
import unittest


try:
    import src.mlstatpy as thismodule
except ImportError:
    path = os.path.normpath(
        os.path.abspath(
            os.path.join(
                os.path.split(__file__)[0],
                "..",
                "..")))
    if path not in sys.path:
        sys.path.append(path)
    import src.mlstatpy as thismodule

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

from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import get_temp_folder, add_missing_development_version
from pyquickhelper.ipythonhelper import execute_notebook_list, execute_notebook_list_finalize_ut, get_additional_paths


class TestNotebook123Coverage(unittest.TestCase):

    def setUp(self):
        add_missing_development_version(["pyensae", "jyquickhelper"],
                                        __file__, hide=True)

    def a_test_notebook_runner(self, name, folder, valid=None):
        temp = get_temp_folder(__file__, "temp_notebook_123_{0}".format(name))
        doc = os.path.join(temp, "..", "..", "..", "_doc", "notebooks", folder)
        self.assertTrue(os.path.exists(doc))
        keepnote = [os.path.join(doc, _) for _ in os.listdir(doc) if name in _]
        self.assertTrue(len(keepnote) > 0)

        import pyquickhelper
        import jyquickhelper
        import pyensae
        add_path = get_additional_paths(
            [jyquickhelper, pyquickhelper, pyensae, thismodule])
        res = execute_notebook_list(
            temp, keepnote, additional_path=add_path, valid=valid)
        execute_notebook_list_finalize_ut(res, fLOG=fLOG, dump=thismodule)

    def test_notebook_roc(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        self.a_test_notebook_runner("roc", "metric")


if __name__ == "__main__":
    unittest.main()
