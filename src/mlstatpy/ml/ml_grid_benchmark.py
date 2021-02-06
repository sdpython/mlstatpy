# -*- coding: utf-8 -*-
"""
@file
@brief About Machine Learning Benchmark
"""
import os
import numpy
from sklearn.model_selection import train_test_split
from sklearn.base import ClusterMixin
from sklearn.metrics import silhouette_score
from pyquickhelper.loghelper import noLOG
from pyquickhelper.benchhelper import GridBenchMark


class MlGridBenchMark(GridBenchMark):
    """
    The class tests a list of model over a list of datasets.
    """

    def __init__(self, name, datasets, clog=None, fLOG=noLOG, path_to_images=".",
                 cache_file=None, progressbar=None, graphx=None, graphy=None,
                 **params):
        """
        @param      name            name of the test
        @param      datasets        list of dictionary of dataframes
        @param      clog            see @see cl CustomLog or string
        @param      fLOG            logging function
        @param      params          extra parameters
        @param      path_to_images  path to images and intermediate results
        @param      cache_file      cache file
        @param      progressbar     relies on *tqdm*, example *tnrange*
        @param      graphx          list of variables to use as X axis
        @param      graphy          list of variables to use as Y axis

        If *cache_file* is specified, the class will store the results of the
        method :meth:`bench <pyquickhelper.benchhelper.benchmark.GridBenchMark.bench>`.
        On a second run, the function load the cache
        and run modified or new run (in *param_list*).

        *datasets* should be a dictionary with dataframes a values
        with the following keys:

        * ``'X'``: features
        * ``'Y'``: labels (optional)
        """
        GridBenchMark.__init__(self, name=name, datasets=datasets, clog=clog, fLOG=fLOG,
                               path_to_images=path_to_images, cache_file=cache_file,
                               progressbar=progressbar, **params)
        self._xaxis = graphx
        self._yaxis = graphy

    def preprocess_dataset(self, dsi, **params):
        """
        Splits the dataset into train and test.

        @param      dsi         dataset index
        @param      params      additional parameters
        @return                 dataset (like info), dictionary for metrics
        """
        ds, appe, params = GridBenchMark.preprocess_dataset(
            self, dsi, **params)

        no_split = ds["no_split"] if "no_split" in ds else False

        if no_split:
            self.fLOG("[MlGridBenchMark.preprocess_dataset] no split")
            return (ds, ds), appe, params

        self.fLOG("[MlGridBenchMark.preprocess_dataset] split train test")
        spl = ["X", "Y", "weight", "group"]
        names = [_ for _ in spl if _ in ds]
        if len(names) == 0:
            raise ValueError(  # pragma: no cover
                "No dataframe or matrix was found.")
        mats = [ds[_] for _ in names]

        pars = {"train_size", "test_size"}
        options = {k: v for k, v in params.items() if k in pars}
        for k in pars:
            if k in params:
                del params[k]

        res = train_test_split(*mats, **options)

        train = {}
        for i, n in enumerate(names):
            train[n] = res[i * 2]
        test = {}
        for i, n in enumerate(names):
            test[n] = res[i * 2 + 1]

        self.fLOG("[MlGridBenchMark.preprocess_dataset] done")
        return (train, test), appe, params

    def bench_experiment(self, ds, **params):
        """
        Calls meth *fit*.
        """
        if not isinstance(ds, tuple) and len(ds) != 2:
            raise TypeError(  # pragma: no cover
                "ds must a tuple with two dictionaries train, test")
        if "model" not in params:
            raise KeyError(  # pragma: no cover
                "params must contains key 'model'")
        model = params["model"]
        # we assume model is a function which creates a model
        model = model()
        del params["model"]
        return self.fit(ds[0], model, **params)

    def predict_score_experiment(self, ds, model, **params):
        """
        Calls method *score*.
        """
        if not isinstance(ds, tuple) and len(ds) != 2:
            raise TypeError(  # pragma: no cover
                "ds must a tuple with two dictionaries train, test")
        if "model" in params:
            raise KeyError(  # pragma: no cover
                "params must not contains key 'model'")
        return self.score(ds[1], model, **params)

    def fit(self, ds, model, **params):
        """
        Trains a model.

        @param      ds          dictionary with the data to use for training
        @param      model       model to train
        """
        if "X" not in ds:
            raise KeyError(  # pragma: no cover
                "ds must contain key 'X'")
        if "model" in params:
            raise KeyError(  # pragma: no cover
                "params must not contain key 'model', this is the model to train")
        X = ds["X"]
        Y = ds.get("Y", None)
        weight = ds.get("weight", None)
        self.fLOG("[MlGridBenchMark.fit] fit", params)

        train_params = params.get("train_params", {})

        if weight is not None:
            model.fit(X=X, y=Y, weight=weight, **train_params)
        else:
            model.fit(X=X, y=Y, **train_params)
        self.fLOG("[MlGridBenchMark.fit] Done.")
        return model

    def score(self, ds, model, **params):
        """
        Scores a model.
        """
        X = ds["X"]
        Y = ds.get("Y", None)

        if "weight" in ds:
            raise NotImplementedError(  # pragma: no cover
                "weight are not used yet")

        metrics = {}
        appe = {}

        if hasattr(model, "score"):
            score = model.score(X, Y)
            metrics["own_score"] = score

        if isinstance(model, ClusterMixin):
            # add silhouette
            if hasattr(model, "predict"):
                ypred = model.predict(X)
            elif hasattr(model, "transform"):
                ypred = model.transform(X)
            elif hasattr(model, "labels_"):
                ypred = model.labels_
            if len(ypred.shape) > 1 and ypred.shape[1] > 1:
                ypred = numpy.argmax(ypred, axis=1)
            score = silhouette_score(X, ypred)
            metrics["silhouette"] = score

        return metrics, appe

    def end(self):
        """
        nothing to do
        """
        pass

    def graphs(self, path_to_images):
        """
        Plots multiples graphs.

        @param      path_to_images  where to store images
        @return     list of tuple (image_name, function to create the graph)
        """
        import matplotlib.pyplot as plt  # pylint: disable=C0415
        import matplotlib.cm as mcm  # pylint: disable=C0415
        df = self.to_df()

        def local_graph(vx, vy, ax=None, text=True, figsize=(5, 5)):
            btrys = set(df["_btry"])
            ymin = df[vy].min()
            ymax = df[vy].max()
            decy = (ymax - ymin) / 50
            colors = mcm.rainbow(numpy.linspace(0, 1, len(btrys)))
            if len(btrys) == 0:
                raise ValueError("The benchmark is empty.")  # pragma: no cover
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=figsize)  # pragma: no cover
                ax.grid(True)  # pragma: no cover
            for i, btry in enumerate(sorted(btrys)):
                subset = df[df["_btry"] == btry]
                if subset.shape[0] > 0:
                    tx = subset[vx].mean()
                    ty = subset[vy].mean()
                    if not numpy.isnan(tx) and not numpy.isnan(ty):
                        subset.plot(x=vx, y=vy, kind="scatter",
                                    label=btry, ax=ax, color=colors[i])
                        if text:
                            ax.text(tx, ty + decy, btry, size='small',
                                    color=colors[i], ha='center', va='bottom')
            ax.set_xlabel(vx)
            ax.set_ylabel(vy)
            return ax

        res = []
        if self._xaxis is not None and self._yaxis is not None:
            for vx in self._xaxis:
                for vy in self._yaxis:
                    self.fLOG("Plotting {0} x {1}".format(vx, vy))
                    func_graph = lambda ax=None, text=True, vx=vx, vy=vy, **kwargs: \
                        local_graph(vx, vy, ax=ax, text=text, **kwargs)

                    if path_to_images is not None:
                        img = os.path.join(
                            path_to_images, "img-{0}-{1}x{2}.png".format(self.Name, vx, vy))
                        gr = self.LocalGraph(
                            func_graph, img, root=path_to_images)
                        self.fLOG("Saving '{0}'".format(img))
                        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                        gr.plot(ax=ax, text=True)
                        fig.savefig(img)
                        self.fLOG("Done")
                        res.append(gr)
                        plt.close('all')
                    else:
                        gr = self.LocalGraph(func_graph)
                        res.append(gr)
        return res

    def plot_graphs(self, grid=None, text=True, **kwargs):
        """
        Plots all graphs in the same graphs.

        @param      grid        grid of axes
        @param      text        add legend title on the graph
        @return                 grid
        """
        nb = len(self.Graphs)
        if nb == 0:
            raise ValueError("No graph to plot.")  # pragma: no cover

        nb = len(self.Graphs)
        if nb % 2 == 0:
            size = nb // 2, 2
        else:
            size = nb // 2 + 1, 2

        if grid is None:
            import matplotlib.pyplot as plt  # pylint: disable=C0415
            fg = kwargs.get('figsize', (5 * size[0], 10))
            _, grid = plt.subplots(size[0], size[1], figsize=fg)
            if 'figsize' in kwargs:
                del kwargs['figsize']  # pragma: no cover
        else:
            shape = grid.shape
            if shape[0] * shape[1] < nb:
                raise ValueError(  # pragma: no cover
                    "The graph is not big enough {0} < {1}".format(shape, nb))

        x = 0
        y = 0
        for i, gr in enumerate(self.Graphs):
            self.fLOG("Plot graph {0}/{1}".format(i + 1, nb))
            gr.plot(ax=grid[y, x], text=text, **kwargs)
            x += 1
            if x >= grid.shape[1]:
                x = 0
                y += 1
        return grid
