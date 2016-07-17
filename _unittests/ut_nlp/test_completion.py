#-*- coding: utf-8 -*-
"""
@brief      test log(time=33s)

https://dumps.wikimedia.org/frwiki/latest/frwiki-latest-all-titles.gz
https://dumps.wikimedia.org/frwiki/latest/frwiki-latest-all-titles-in-ns0.gz
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
from src.mlstatpy.nlp.completion import CompletionTrieNode


class TestCompletion(unittest.TestCase):

    def test_build_trie(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        queries = [(1, 'a'), (2, 'ab'), (3, 'abc'), (4, 'abcd'), (5, 'bc')]
        trie = CompletionTrieNode.build(queries)
        res = list(trie.items())
        self.assertEqual(len(res), 2)
        res = list(trie.iter_leaves())
        self.assertEqual(
            res, [(1, 'a'), (2, 'ab'), (3, 'abc'), (4, 'abcd'), (5, 'bc')])
        lea = list(trie.leaves())
        self.assertEqual(len(lea), 5)
        assert all(_.leave for _ in lea)
        node = trie.find('b')
        assert node is not None
        assert not node.leave
        node = trie.find('ab')
        assert node is not None
        assert node.leave
        self.assertEqual(node.value, 'ab')
        for k, word in queries:
            ks = trie.min_keystroke(word)
            self.assertEqual(ks[0], ks[1])

    def test_build_trie_mks(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        queries = [(4, 'a'), (2, 'ab'), (3, 'abc'), (1, 'abcd')]
        trie = CompletionTrieNode.build(queries)
        nodes = trie.items_list()
        st = [str(_) for _ in nodes]
        fLOG(st)
        self.assertEqual(
            st, ['-::1', '#:a:4', '#:ab:2', '#:abc:3', '#:abcd:1'])
        find = trie.find('a')
        assert find
        ms = [(word, trie.min_keystroke(word)) for k, word in queries]
        self.assertEqual(ms, [('a', (1, 1)), ('ab', (2, 2)),
                              ('abc', (3, 3)), ('abcd', (1, 0))])

    def test_build_trie_mks_min(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        queries = [(None, 'a'), (None, 'ab'), (None, 'abc'), (None, 'abcd')]
        trie = CompletionTrieNode.build(queries)
        gain = sum(len(w) - trie.min_keystroke(w)[0] for a, w in queries)
        self.assertEqual(gain, 0)
        for per in itertools.permutations(queries):
            trie = CompletionTrieNode.build(per)
            gain = sum(len(w) - trie.min_keystroke(w)[0] for a, w in per)
            fLOG(gain, per)

    def test_build_dynamic_trie_mks_min(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        queries = [(None, 'a'), (None, 'ab'), (None, 'abc'), (None, 'abcd')]
        trie = CompletionTrieNode.build(queries)
        trie.precompute_stat()
        for leave in trie.leaves():
            if leave.stat is None:
                raise Exception("None for {0}".format(leave))
            mk = trie.min_keystroke(leave.value)
            fLOG(leave.value, mk, leave.stat.str_mks())
            self.assertEqual(mk, (leave.stat.mks0, leave.stat.mks0_))


if __name__ == "__main__":
    unittest.main()
