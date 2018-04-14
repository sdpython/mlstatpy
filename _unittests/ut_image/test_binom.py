# -*- coding: utf-8 -*-
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
from src.mlstatpy.image.detection_segment import tabule_queue_binom


class TestQueueBinom(unittest.TestCase):

    def test_queue(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        b = tabule_queue_binom(2, 2)
        self.assertEqual(b, {(0, 1): 0.0, (1, 2): 0.0, (0, 0): 1.0, (2, 3): 0.0,
                             (2, 0): 1.0, (1, 0): 1.0, (2, 2): 4.0, (1, 1): 2.0, (2, 1): 0.0})


if __name__ == "__main__":
    unittest.main()
