# -*- coding: utf-8 -*-
"""
@brief      test log(time=60s)
"""

import sys
import os
import unittest
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import get_temp_folder, ExtTestCase


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

from src.mlstatpy.data.wikipedia import download_titles


class TestLONGWikipediaPageCount(ExtTestCase):

    def test_wikipedia_page_count(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        temp = get_temp_folder(__file__, "temp_wikipedia_title")
        name = download_titles("fr", folder=temp, fLOG=fLOG)
        self.assertLesser(len(name), 2000)
        self.assertExists(name)


if __name__ == "__main__":
    unittest.main()
