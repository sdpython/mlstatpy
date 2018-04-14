# -*- coding: utf-8 -*-
"""
@brief      test log(time=4s)
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
from src.mlstatpy.garden.poulet import maximum, find_maximum, histogramme_poisson_melange, proba_poisson_melange


class TestPoulet(unittest.TestCase):

    def test_poulet1(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        res = maximum(2, 5, 1, 80)
        m = find_maximum(res)
        self.assertEqual(m, (86, 228.50205712688214))
        self.assertEqual(
            res[:3], [(0, 0.0), (1, 2.9999999999999942), (2, 5.9999999999999885)])

    def test_poulet2(self):

        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")
        h = histogramme_poisson_melange([48, 10, 4], [1, 2, 3])
        self.assertTrue(max(h) > 0.01)
        self.assertEqual(h[:4], [0.0, 0.0, 0.0, 0.0])

    def test_poulet3(self):

        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")
        h = proba_poisson_melange([48, 10, 4], [1, 2, 3], 20)
        self.assertEqual(h, 0)
        h = proba_poisson_melange([48, 10, 4], [1, 2, 3], 40)
        self.assertTrue(h < 0.1)


if __name__ == "__main__":
    unittest.main()
