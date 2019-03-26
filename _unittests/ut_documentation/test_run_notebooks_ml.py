# -*- coding: utf-8 -*-
"""
@brief      test log(time=33s)
"""
import os
import unittest
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import get_temp_folder
from pyquickhelper.ipythonhelper import execute_notebook_list, execute_notebook_list_finalize_ut
import mlstatpy


class TestRunNotebooksML(unittest.TestCase):

    def test_run_notebook_ml(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        temp = get_temp_folder(__file__, "temp_run_notebooks_ml")

        # selection of notebooks
        fnb = os.path.normpath(os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "..", "..", "_doc", "notebooks", "ml"))
        keepnote = []
        for f in os.listdir(fnb):
            if os.path.splitext(f)[-1] == ".ipynb" and "_long" not in f:
                keepnote.append(os.path.join(fnb, f))

        # function to tell that a can be run
        def valid(cell):
            if "open_html_form" in cell:
                return False
            if "open_window_params" in cell:
                return False
            if '<div style="position:absolute' in cell:
                return False
            return True

        # additionnal path to add
        addpaths = [os.path.normpath(os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "..", "..", "src")),
            os.path.normpath(os.path.join(
                os.path.abspath(os.path.dirname(__file__)), "..", "..", "..", "pyquickhelper", "src")),
            os.path.normpath(os.path.join(
                os.path.abspath(os.path.dirname(__file__)), "..", "..", "..", "jyquickhelper", "src"))
        ]

        # run the notebooks
        res = execute_notebook_list(
            temp, keepnote, fLOG=fLOG, valid=valid, additional_path=addpaths)
        execute_notebook_list_finalize_ut(
            res, fLOG=fLOG, dump=mlstatpy)


if __name__ == "__main__":
    unittest.main()
