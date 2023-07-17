# -*- coding: utf-8 -*-
"""
@brief      test log(time=312s)
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


class TestRunNotebooksML(unittest.TestCase):
    def setUp(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")

        add_missing_development_version(
            ["pyensae", "jyquickhelper"], __file__, hide=True
        )

    def a_test_notebook_runner(self, name, folder, valid=None):
        fLOG(f"notebook {name!r} in {folder!r}")
        temp = get_temp_folder(__file__, f"temp_notebook_ml_{name}")
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

    def test_notebook_benchmark(self):
        self.a_test_notebook_runner("benchmark", "ml")

    def test_notebook_logreg_voronoi(self):
        self.a_test_notebook_runner("logreg_voronoi", "ml")

    def test_notebook_mf_acp(self):
        self.a_test_notebook_runner("mf_acp", "ml")

    def test_notebook_neural_tree(self):
        self.a_test_notebook_runner("neural_tree", "ml")

    def test_notebook_piecewise_linear_regression(self):
        self.a_test_notebook_runner("piecewise_linear_regression", "ml")

    def test_notebook_regression_no_inversion(self):
        self.a_test_notebook_runner("regression_no_inversion", "ml")

    def test_notebook_valeurs_manquantes_mf(self):
        self.a_test_notebook_runner("valeurs_manquantes_mf", "ml")

    def test_notebook_reseau_neurones(self):
        self.a_test_notebook_runner("reseau_neurones", "ml")

    def test_notebook_survival(self):
        self.a_test_notebook_runner("survival", "ml")


if __name__ == "__main__":
    unittest.main()
