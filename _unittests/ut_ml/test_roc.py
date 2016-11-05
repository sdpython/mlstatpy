#-*- coding: utf-8 -*-
"""
@brief      test log(time=38s)
"""

import sys
import os
import unittest
import random
import matplotlib.pyplot as plt


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
from pyquickhelper.pycode import get_temp_folder
from src.mlstatpy.ml.roc import ROC


class TestROC(unittest.TestCase):

    def test_roc(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        temp = get_temp_folder(__file__, "temp_roc")

        data = [random.random() for a in range(0, 1000)]
        data = [(x, 1 if x + random.random() / 3 > 0.7 else 0) for x in data]

        test = ROC(data)
        fLOG(test.__str__())
        roc = test.compute_roc_curve()
        t = test.roc_intersect(roc, 0.2)
        assert 1 >= t >= 0

        fig, ax = plt.subplots()
        ax = test.plot(10, ax=ax, label="r10")
        assert ax
        fig.savefig(os.path.join(temp, "roc10.png"))
        test.plot(100, ax=ax, label="r100")
        assert ax
        fig.savefig(os.path.join(temp, "roc100.png"))
        ax = test.plot(100, ax=ax, bootstrap=10)
        assert ax
        fig.savefig(os.path.join(temp, "roc100b.png"))

        fLOG("computing rate..............................")
        values = test.auc_interval(alpha=0.1, bootstrap=20)
        for k, v in sorted(values.items()):
            fLOG("{0}={1}".format(k, v))
        self.assertEqual(list(sorted(values.keys())), [
                         'auc', 'interval', 'max', 'mean', 'mediane', 'min', 'var'])
        assert values["min"] <= values["auc"] <= values["max"]

        fLOG("computing rate..............................")
        values = test.roc_intersect_interval(
            0.1, 100, True, bootstrap=50)
        for k, v in sorted(values.items()):
            fLOG("{0}={1}".format(k, v))
        self.assertEqual(list(sorted(values.keys())), [
                         'interval', 'max', 'mean', 'mediane', 'min', 'var', 'y'])
        assert values["min"] <= values["y"] <= values["max"]


if __name__ == "__main__":
    unittest.main()
