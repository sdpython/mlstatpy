"""
@brief      test log(time=20s)
@author     Xavier Dupre
"""
import os
import unittest
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import get_temp_folder, fix_tkinter_issues_virtualenv
from mlstatpy.ml import MlGridBenchMark


class TestMlGridBenchMark(unittest.TestCase):

    def test_ml_benchmark(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        fix_tkinter_issues_virtualenv(fLOG=fLOG)
        import matplotlib.pyplot as plt  # pylint: disable=C0415
        try:
            import dill  # pylint: disable=C0415
            pickle_module = dill
        except ImportError:
            try:
                import cloudpickle
                pickle_module = cloudpickle
            except ImportError:
                raise ImportError(  # pylint: disable=W0707
                    "Unable to import 'dill' or 'cloudpickle'. Cannot pickle a lambda function.")

        self.assertTrue(plt is not None)
        temp = get_temp_folder(__file__, "temp_ml_grid_benchmark")

        params = [dict(model=lambda: KMeans(n_clusters=3), name="KMeans-3",
                       shortname="km-3"),
                  dict(model=lambda: AgglomerativeClustering(),
                       name="AgglomerativeClustering", shortname="aggclus")]

        datasets = [dict(X=make_blobs(100, centers=3)[0], Nclus=3,
                         name="blob-100-3", shortname="b-100-3", no_split=True),
                    dict(X=make_blobs(100, centers=5)[0], Nclus=5,
                         name="blob-100-5", shortname="b-100-5", no_split=True)]

        for no_split in [True, False]:
            bench = MlGridBenchMark(
                "TestName", datasets, fLOG=fLOG, clog=temp, path_to_images=temp,
                cache_file=os.path.join(temp, "cache.pickle"),
                pickle_module=pickle_module, repetition=3,
                graphx=["_time", "time_train", "Nclus"],
                graphy=["silhouette", "Nrows"],
                no_split=no_split)
            bench.run(params)
            spl = bench.preprocess_dataset(0)
            self.assertIsInstance(spl, tuple)
            self.assertEqual(len(spl), 3)
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
