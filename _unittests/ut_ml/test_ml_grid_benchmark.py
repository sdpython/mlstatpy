"""
@brief      test log(time=2s)
@author     Xavier Dupre
"""

import sys
import os
import unittest


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

from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import get_temp_folder
from src.mlstatpy.ml import MlGridBenchMark
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs


class TestMlGridBenchMark(unittest.TestCase):

    def test_ml_benchmark(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        fix_tkinter_issues_virtualenv(fLOG=fLOG)
        import matplotlib.pyplot as plt
        import dill
        self.assertTrue(plt is not None)
        temp = get_temp_folder(__file__, "temp_ml_grid_benchmark")

        params = [dict(model=lambda: KMeans(n_clusters=3), name="KMeans-3", shortname="km-3"),
                  dict(model=lambda: AgglomerativeClustering(), name="AgglomerativeClustering", shortname="aggclus")]

        datasets = [dict(X=make_blobs(100, centers=3)[0], Nclus=3,
                         name="blob-100-3", shortname="b-100-3", no_split=True),
                    dict(X=make_blobs(100, centers=5)[0], Nclus=5,
                         name="blob-100-5", shortname="b-100-5", no_split=True)]

        bench = MlGridBenchMark("TestName", datasets, fLOG=fLOG, clog=temp,
                                path_to_images=temp,
                                cache_file=os.path.join(temp, "cache.pickle"),
                                pickle_module=dill, repetition=3,
                                graphx=["_time", "time_train", "Nclus"],
                                graphy=["silhouette", "Nrows"])
        bench.run(params)
        df = bench.to_df()
        ht = df.to_html(float_format="%1.3f", index=False)
        self.assertTrue(len(df) > 0)
        self.assertIsNotNone(ht)
        self.assertEqual(df.shape[0], 12)
        report = os.path.join(temp, "report.html")
        csv = os.path.join(temp, "report.csv")
        rst = os.path.join(temp, "report.rst")
        bench.report(filehtml=report, filecsv=csv, filerst=rst,
                     title="A Title", description="description")
        self.assertTrue(os.path.exists(report))
        self.assertTrue(os.path.exists(csv))
        self.assertTrue(os.path.exists(rst))

        graph = bench.plot_graphs()
        self.assertIsNotNone(graph)
        self.assertEqual(graph.shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
