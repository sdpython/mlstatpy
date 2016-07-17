#-*- coding: utf-8 -*-
"""
@brief      test log(time=33s)
"""

import sys
import os
import unittest
import random


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
from src.mlstatpy.ml.roc import ROC


class TestROC(unittest.TestCase):

    def test_roc(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        data = [random.random() for a in range(0, 1000)]
        data = [(x, 1 if x + random.random() / 50 > 0.7 else 0) for x in data]

        test = ROC(data)
        fLOG(test.__str__())

        ax = test.plot([1000])
        assert ax
        ax = test.plot([10, 100, 1000, 5000])
        assert ax
        return
        fLOG("computing rate..............................")
        rate, inte, mmm = test.ROC_point_intervalle(
            0.1, 100, read=True, bootstrap=50)
        fLOG("rate = \t", "%3.2f" % (rate * 100), "%")
        fLOG("intervalle à 95% = \t", "[%3.2f, %3.2f]" % (
            inte[0] * 100, inte[1] * 100))
        fLOG("intervalle min,max = \t", "[%3.2f, %3.2f]" % (
            mmm[0] * 100, mmm[1] * 100))
        fLOG("moyenne = %3.2f, écart-type = %3.2f, médiance = %3.2f" %
             (mmm[2] * 100, mmm[3] * 100, mmm[4] * 100))

        rate, inte, mmm = test.ROC_AUC(0.1, 100, bootstrap=200)
        fLOG("AUC= \t", "%3.2f" % (rate))
        fLOG("intervalle à 95% = \t", "[%3.2f, %3.2f]" % (inte[0], inte[1]))
        fLOG("intervalle min,max = \t", "[%3.2f, %3.2f]" % (mmm[0], mmm[1]))
        fLOG("moyenne = %3.2f, écart-type = %3.2f, médiance = %3.2f" %
             (mmm[2] * 100, mmm[3] * 100, mmm[4] * 100))

        test.DrawROC([100], read=True, bootstrap=100)

if __name__ == "__main__":
    unittest.main()
