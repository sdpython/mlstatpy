import os
import unittest
import random
from mlstatpy.ext_test_case import get_temp_folder, ExtTestCase
from mlstatpy.ml.roc import ROC


class TestROC(ExtTestCase):
    def test_roc(self):
        import matplotlib.pyplot as plt

        temp = get_temp_folder(__file__, "temp_roc")

        data = [random.random() for a in range(1000)]
        data = [(x, 1 if x + random.random() / 3 > 0.7 else 0) for x in data]

        test = ROC(y_true=[_[1] for _ in data], y_score=[_[0] for _ in data])
        self.assertNotEmpty(test.Data)
        self.assertNotEmpty(repr(test))
        self.assertEqual(len(test), len(data))
        test = ROC(df=data)

        roc = test.compute_roc_curve()
        t = test.roc_intersect(roc, 0.2)
        self.assertTrue(1 >= t >= 0)

        conf = test.confusion()
        str(conf)

        self.assertEqual(conf.shape, (12, 5))
        conf = test.confusion(score=0.5)

        self.assertEqual(conf.shape, (1, 5))

        fig, ax = plt.subplots()
        ax = test.plot(0, ax=ax, curve=ROC.CurveType.PROBSCORE, thresholds=True)
        self.assertNotEmpty(ax)
        fig.savefig(os.path.join(temp, "roc_PROBSCORE_10.png"))

        fig, ax = plt.subplots()
        test.plot(
            0, ax=ax, bootstrap=10, curve=ROC.CurveType.PROBSCORE, thresholds=True
        )
        self.assertNotEmpty(ax)
        fig.savefig(os.path.join(temp, "roc_PROBSCORE_100_b10.png"))

        fig, ax = plt.subplots()
        ax = test.plot(0, ax=ax, curve=ROC.CurveType.SKROC)
        self.assertNotEmpty(ax)
        fig.savefig(os.path.join(temp, "roc_SKROC_10.png"))

        fig, ax = plt.subplots()
        test.plot(0, ax=ax, bootstrap=10, curve=ROC.CurveType.SKROC)
        self.assertNotEmpty(ax)
        fig.savefig(os.path.join(temp, "roc_SKROC_100_b10.png"))

        fig, ax = plt.subplots()
        ax = test.plot(100, ax=ax, curve=ROC.CurveType.RECPREC)
        self.assertNotEmpty(ax)
        fig.savefig(os.path.join(temp, "roc_RECPREC_100.png"))

        fig, ax = plt.subplots()
        ax = test.plot(100, ax=ax, bootstrap=10, curve=ROC.CurveType.RECPREC)
        self.assertNotEmpty(ax)
        fig.savefig(os.path.join(temp, "roc_RECPREC_100_b10.png"))

        fig, ax = plt.subplots()
        ax = test.plot(0, ax=ax, curve=ROC.CurveType.SKROC, thresholds=True)
        self.assertNotEmpty(ax)
        fig.savefig(os.path.join(temp, "roc_SKROC_T_10.png"))

        fig, ax = plt.subplots()
        test.plot(0, ax=ax, bootstrap=10, curve=ROC.CurveType.SKROC, thresholds=True)
        self.assertNotEmpty(ax)
        fig.savefig(os.path.join(temp, "roc_SKROC_T_100_b10.png"))

        fig, ax = plt.subplots()
        ax = test.plot(100, ax=ax, curve=ROC.CurveType.RECPREC, thresholds=True)
        self.assertNotEmpty(ax)
        fig.savefig(os.path.join(temp, "roc_RECPREC_T_100.png"))

        fig, ax = plt.subplots()
        ax = test.plot(
            100, ax=ax, bootstrap=10, curve=ROC.CurveType.RECPREC, thresholds=True
        )
        self.assertNotEmpty(ax)
        fig.savefig(os.path.join(temp, "roc_RECPREC_T_100_b10.png"))

        fig, ax = plt.subplots()
        ax = test.plot(100, ax=ax, curve=ROC.CurveType.ERRREC)
        self.assertNotEmpty(ax)
        fig.savefig(os.path.join(temp, "roc_ERRREC_100.png"))

        fig, ax = plt.subplots()
        ax = test.plot(100, ax=ax, bootstrap=10, curve=ROC.CurveType.ERRREC)
        self.assertNotEmpty(ax)
        fig.savefig(os.path.join(temp, "roc_ERRREC_100_b10.png"))

        fig, ax = plt.subplots()
        self.assertRaise(
            lambda: test.plot(10, ax=ax, label=["r10", "p10"], curve=ROC.CurveType.ROC),
            ValueError,
        )
        ax = test.plot(
            10, ax=ax, thresholds=True, label=["r10", "p10"], curve=ROC.CurveType.ROC
        )
        self.assertNotEmpty(ax)
        fig.savefig(os.path.join(temp, "roc_ROC_10.png"))

        fig, ax = plt.subplots()
        test.plot(
            100, ax=ax, label=["r100", "p100"], curve=ROC.CurveType.ROC, thresholds=True
        )
        self.assertNotEmpty(ax)
        fig.savefig(os.path.join(temp, "roc_ROC_100.png"))

        fig, ax = plt.subplots()
        test.plot(100, ax=ax, bootstrap=10, curve=ROC.CurveType.ROC)
        self.assertNotEmpty(ax)
        fig.savefig(os.path.join(temp, "roc_ROC_100_b10.png"))

        values = test.auc_interval(alpha=0.1, bootstrap=20)
        self.assertEqual(
            list(sorted(values.keys())),
            ["auc", "interval", "max", "mean", "mediane", "min", "var"],
        )
        self.assertTrue(values["min"] <= values["auc"] <= values["max"])

        values = test.roc_intersect_interval(0.1, 100, bootstrap=50)
        self.assertEqual(
            list(sorted(values.keys())),
            ["interval", "max", "mean", "mediane", "min", "var", "y"],
        )
        self.assertTrue(values["min"] <= values["y"] <= values["max"])
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
