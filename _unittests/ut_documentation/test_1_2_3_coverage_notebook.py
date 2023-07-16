# -*- coding: utf-8 -*-
"""
@brief      test log(time=21s)
"""
import os
import unittest
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import get_temp_folder, add_missing_development_version
from pyquickhelper.ipythonhelper import (
    execute_notebook_list,
    execute_notebook_list_finalize_ut,
    get_additional_paths,
)
import mlstatpy as thismodule


class TestNotebook123Coverage(unittest.TestCase):
    def setUp(self):
        add_missing_development_version(
            ["pyensae", "jyquickhelper"], __file__, hide=True
        )

    def a_test_notebook_runner(self, name, folder, valid=None):
        temp = get_temp_folder(__file__, f"temp_notebook_123_{name}")
        doc = os.path.join(temp, "..", "..", "..", "_doc", "notebooks", folder)
        self.assertTrue(os.path.exists(doc))
        keepnote = [os.path.join(doc, _) for _ in os.listdir(doc) if name in _]
        self.assertTrue(len(keepnote) > 0)

        import pyquickhelper  # pylint: disable=C0415
        import jyquickhelper  # pylint: disable=C0415
        import pyensae  # pylint: disable=C0415

        add_path = get_additional_paths(
            [jyquickhelper, pyquickhelper, pyensae, thismodule]
        )
        res = execute_notebook_list(
            temp, keepnote, additional_path=add_path, valid=valid
        )
        execute_notebook_list_finalize_ut(res, fLOG=fLOG, dump=thismodule)

    def test_notebook_roc(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")

        self.a_test_notebook_runner("roc", "metric")

    def test_notebook_pvalue(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")

        self.a_test_notebook_runner("pvalue", "metric")


if __name__ == "__main__":
    unittest.main()
