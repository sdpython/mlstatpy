# -*- coding: utf-8 -*-
"""
@brief      test log(time=33s)
"""
import os
import unittest
import shutil
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import get_temp_folder
from pyquickhelper.ipythonhelper import (
    execute_notebook_list,
    execute_notebook_list_finalize_ut,
)
import mlstatpy


class TestRunNotebooksImage(unittest.TestCase):
    def test_run_notebook(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")

        temp = get_temp_folder(__file__, "temp_run_notebooks_image")

        # selection of notebooks
        fnb = os.path.normpath(
            os.path.join(
                os.path.abspath(os.path.dirname(__file__)),
                "..",
                "..",
                "_doc",
                "notebooks",
                "image",
            )
        )
        keepnote = []
        for f in os.listdir(fnb):
            if os.path.splitext(f)[-1] == ".ipynb" and "long" not in f:
                keepnote.append(os.path.join(fnb, f))

        # function to tell that a can be run
        def valid(cell):
            return True

        # additionnal path to add
        addpaths = [
            os.path.normpath(
                os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "..")
            ),
        ]

        # copy
        shutil.copy(os.path.join(fnb, "eglise_zoom2.jpg"), temp)

        # run the notebooks
        res = execute_notebook_list(
            temp, keepnote, fLOG=fLOG, valid=valid, additional_path=addpaths
        )
        execute_notebook_list_finalize_ut(res, fLOG=fLOG, dump=mlstatpy)


if __name__ == "__main__":
    unittest.main()
