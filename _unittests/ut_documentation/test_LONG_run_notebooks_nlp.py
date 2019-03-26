# -*- coding: utf-8 -*-
"""
@brief      test log(time=571s)
"""
import os
import unittest
from pyquickhelper.loghelper import fLOG, CustomLog
from pyquickhelper.pycode import get_temp_folder, add_missing_development_version
from pyquickhelper.ipythonhelper import execute_notebook_list, execute_notebook_list_finalize_ut
import mlstatpy


class TestLONGRunNotebooksNLP(unittest.TestCase):

    def setUp(self):
        add_missing_development_version(["pymyinstall", "pyensae", "pymmails", "jyquickhelper"],
                                        __file__, hide=True)

    def test_long_run_notebook(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        temp = get_temp_folder(__file__, "temp_run_notebooks_nlp")

        # selection of notebooks
        fnb = os.path.normpath(os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "..", "..", "_doc", "notebooks", "nlp"))
        keepnote = []
        for f in os.listdir(fnb):
            if os.path.splitext(f)[-1] == ".ipynb" and "_long" in f:
                keepnote.append(os.path.join(fnb, f))

        # function to tell that a can be run
        def valid(cell):
            if "open_html_form" in cell:
                return False
            if "open_html_form" in cell:
                return False
            if "[50000, 100000, 200000, 500000, 500000, 1000000, 2000000, None]" in cell:
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
        clog = CustomLog(temp)
        clog("START")
        res = execute_notebook_list(temp, keepnote, fLOG=clog, deepfLOG=clog, valid=valid,
                                    additional_path=addpaths)
        clog("END")
        execute_notebook_list_finalize_ut(
            res, fLOG=fLOG, dump=mlstatpy)


if __name__ == "__main__":
    unittest.main()
