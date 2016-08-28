#-*- coding: utf-8 -*-
"""
@brief      test log(time=3s)
"""

import sys
import os
import unittest
import itertools


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


class TestCompletionSimple(unittest.TestCase):

    def test_build_trie_simple(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        queries = [(1, 'a'), (2, 'ab'), (3, 'abc'), (4, 'abcd'), (5, 'bc')]
        trie = CompletionSystem(queries)
        res = list(trie.items())
        self.assertEqual(len(res), 5)
        res = list(trie.tuples())
        self.assertEqual(
            res, [(1, 'a'), (2, 'ab'), (3, 'abc'), (4, 'abcd'), (5, 'bc')])
        node = trie.find('b')
        assert node is None
        node = trie.find('ab')
        assert node is not None
        self.assertEqual(node.value, 'ab')
        trie.compute_metrics(fLOG=fLOG)
        for el in trie:
            self.assertEqual(el.mks0, el.mks1)
            self.assertEqual(el.mks0, el.mks2)
            s = el.str_mks()
            assert s is not None
            # fLOG(s, el.value)
        diffs = trie.compare_with_trie()
        if diffs:
            res = [_[-1] for _ in diffs]
            raise Exception("\n".join(res))

    def test_permutations(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        queries = ['actuellement', 'actualité', 'actu']
        weights = [1, 1, 0]
        for per in itertools.permutations(zip(queries, weights)):
            trie = CompletionSystem([(None, w) for w, p in per])
            trie.compute_metrics()
            # fLOG("----", per)
            for n in trie:
                assert n.mks1 <= n.mks0
            diffs = trie.compare_with_trie()
            if diffs:
                res = [_[-1] for _ in diffs]
                raise Exception("\n".join(res))

    def test_mks_consistency(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        titles = [(None, '"contra el gang del chicharron"',
                   '"Contra el gang del chicharron')]
        trie = CompletionSystem(titles)
        diffs = trie.compare_with_trie()
        if diffs:
            res = [_[-1] for _ in diffs]
            raise Exception("\n".join(res))

        titles.append((None, '"la sequestree"', '"La séquestrée'))
        trie = CompletionSystem(titles)
        diffs = trie.compare_with_trie()
        if diffs:
            res = [_[-1] for _ in diffs]
            raise Exception("\n".join(res))

    def test_completions(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        this = os.path.abspath(os.path.dirname(__file__))
        data = os.path.join(this, "data", "sample300.txt")
        with open(data, "r", encoding="utf-8") as f:
            lines = [_.strip(" \n\r\t") for _ in f.readlines()]

        trie = CompletionSystem([(None, q) for q in lines])
        diffs = trie.compare_with_trie()
        if diffs:
            res = [_[-1] for _ in diffs]
            raise Exception("\n".join(res))

if __name__ == "__main__":
    unittest.main()
