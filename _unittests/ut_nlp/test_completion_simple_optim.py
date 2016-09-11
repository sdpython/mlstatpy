#-*- coding: utf-8 -*-
"""
@brief      test log(time=3s)
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
from src.mlstatpy.nlp.completion_simple import CompletionSystem


class TestCompletionSimpleOptimisation(unittest.TestCase):

    def test_build_trie_simple(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        comp = [(1, 'a'), (2, 'ab'), (3, 'abc'), (4, 'abcd'), (5, 'bc')]
        cset = CompletionSystem(comp)
        cset.compute_metrics(fLOG=fLOG)
        queries = [(q, w) for w, q in comp]
        res = cset.test_metric(queries)
        self.assertEqual(res["mks1"], res["sum_wlen"])
        self.assertEqual(res["hist"]["l"], {1: 1, 2: 7, 3: 3, 4: 4})

        comp = [q for w, q in comp]
        comp.reverse()
        cset = CompletionSystem(comp)
        cset.compute_metrics(fLOG=fLOG)
        queries = [(q, 1) for q in comp]
        for el, found in cset.enumerate_test_metric(queries):
            # fLOG(el, found)
            assert found is not None
        res = cset.test_metric(queries)
        # for k, v in sorted(res.items()): fLOG(k, "=", v)
        assert res["mks1"] < res["sum_wlen"]
        self.assertEqual(res["n"], 5)
        self.assertEqual(res["hist"]["l"], {1: 1, 2: 2, 3: 1, 4: 1})

        # one suggestion in the completion set
        comp = ['a', 'abc', 'bc']
        cset = CompletionSystem(comp)
        cset.compute_metrics(fLOG=fLOG)
        queries = [(q, 1) for q in comp] + [('abcd', 1)]
        for el, found in cset.enumerate_test_metric(queries):
            if found is None:
                fLOG(el.str_mks(), "*", el.value)
            else:
                fLOG(el.str_mks(), "*", el.value, "*", found, found.str_mks())
            fLOG(el.weight)
            self.assertEqual(el.weight, 1)
            if el.value == "abcd":
                assert found is None
                self.assertEqual(el.mks0, 0)
                self.assertEqual(el.mks1, 3)
                self.assertEqual(el.mks2, 3)
            elif el.value == "abc":
                assert found is not None
                self.assertEqual(el.value, found.value)
                self.assertEqual(el.mks0, found.mks0)
                self.assertEqual(el.mks1, found.mks1)
                self.assertEqual(el.mks2, found.mks2)
        res = cset.test_metric(queries)
        for k, v in sorted(res.items()):
            fLOG(k, "=", v)
        assert res["mks1"] < res["sum_wlen"]
        self.assertEqual(res["n"], 4)
        self.assertEqual(res["hist"]["l"], {1: 1, 2: 1, 3: 1, 4: 1})


if __name__ == "__main__":
    unittest.main()
