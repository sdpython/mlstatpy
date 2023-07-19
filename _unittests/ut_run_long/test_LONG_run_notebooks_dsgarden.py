# -*- coding: utf-8 -*-
"""
@brief      test log(time=33s)
"""
import os
import unittest
from pyquickhelper.pycode import get_temp_folder
from pyquickhelper.ipythonhelper import (
    execute_notebook_list,
    execute_notebook_list_finalize_ut,
)
import mlstatpy


class TestRunNotebooksDsGarden(unittest.TestCase):
    def test_run_notebook(self):
        temp = get_temp_folder(__file__, "temp_run_notebooks_dsgarden")

        # selection of notebooks
        fnb = os.path.normpath(
            os.path.join(
                os.path.abspath(os.path.dirname(__file__)),
                "..",
                "..",
                "_doc",
                "notebooks",
                "dsgarden",
            )
        )
        keepnote = []
        for f in os.listdir(fnb):
            if os.path.splitext(f)[-1] == ".ipynb" and "long" not in f:
                keepnote.append(os.path.join(fnb, f))
        self.assertTrue(len(keepnote) > 0)

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
        addpaths = [
            os.path.normpath(
                os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "..")
            ),
        ]

        # run the notebooks
        res = execute_notebook_list(
            temp, keepnote, valid=valid, additional_path=addpaths
        )
        execute_notebook_list_finalize_ut(res, dump=mlstatpy)


if __name__ == "__main__":
    unittest.main()
