"""
@brief      test log(time=2s)
@author     Xavier Dupre
"""

import unittest
from pyquickhelper.pycode import ExtTestCase
from mlstatpy.ml.logreg import criteria, criteria2, random_set_1d, plot_ds


class TestLogReg(ExtTestCase):
    def test_criteria(self):
        for b in [2, 3, 4]:
            with self.subTest(kind=b):
                X, y = random_set_1d(1000, b)
                df = criteria(X, y)
                self.assertEqual(df.shape, (998, 8))
        self.assertRaise(lambda: random_set_1d(1000, 0), ValueError)

    def test_criteria_plot(self):
        X1, y1 = random_set_1d(1000, 2)
        X2, y2 = random_set_1d(1000, 3)
        df1 = criteria(X1, y1)
        df2 = criteria(X2, y2)

        import matplotlib.pyplot as plt

        _, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        plot_ds(X1, y1, ax=ax[0], title="easy")
        plot_ds(X2, y2, ax=ax[1], title="difficult")
        df1.plot(x="X", y=["Gini", "Gain", "p1", "p2"], ax=ax[0], lw=5.0)
        df2.plot(x="X", y=["Gini", "Gain", "p1", "p2"], ax=ax[1], lw=5.0)
        plt.clf()

    def test_criteria2(self):
        X1, y1 = random_set_1d(1000, 2)
        df1 = criteria2(X1, y1)
        self.assertEqual(df1.shape[0], 998)


if __name__ == "__main__":
    unittest.main()
