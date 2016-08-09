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
from src.mlstatpy.nlp.completion import CompletionTrieNode
from src.mlstatpy.data.wikipedia import normalize_wiki_text, enumerate_titles
from src.mlstatpy.nlp.normalize import remove_diacritics


class TestCompletionMks(unittest.TestCase):

    def test_mks_consistency(self):

        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        def cmks(trie):
            trie.precompute_stat()
            trie.update_stat_dynamic()
            gmks = 0.0
            gmksd = 0.0
            gmksd2 = 0.0
            nb = 0
            size = 0
            for n in trie.leaves():
                gmks += len(n.value) - n.stat.mks0
                gmksd += len(n.value) - n.stat.mks
                gmksd2 += len(n.value) - n.stat.mks2
                size += len(n.value)
                nb += 1
                if gmksd2 > gmksd:
                    raise Exception("issue with query '{0}'\n{1}".format(
                        n.value, n.stat.str_mks()))
            return nb, gmks, gmksd, gmksd2, size

        def gain_dynamique_moyen_par_mot(queries, weights):
            per = list(zip(weights, queries))
            total = sum(w * len(q) for q, w in zip(queries, weights))
            res = []
            trie = CompletionTrieNode.build([(None, q) for _, q in per])
            trie.precompute_stat()
            trie.update_stat_dynamic()
            wks = [(w, p, len(w) - trie.min_keystroke0(w)[0]) for p, w in per]
            wks_dyn = [(w, p, len(w) - trie.min_dynamic_keystroke(w)[0])
                       for p, w in per]
            wks_dyn2 = [(w, p, len(w) - trie.min_dynamic_keystroke2(w)[0])
                        for p, w in per]
            gain = sum(g * p / total for w, p, g in wks)
            gain_dyn = sum(g * p / total for w, p, g in wks_dyn)
            gain_dyn2 = sum(g * p / total for w, p, g in wks_dyn2)
            ave_length = sum(len(w) * p / total for p, w in per)
            return gain, gain_dyn, gain_dyn2, ave_length

        this = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "data", "sample_alpha_2.txt"))
        with open(this, "r", encoding="utf-8") as f:
            titles = [_.strip(" \n\r\t") for _ in f.readlines()]
        fLOG(titles[:5])
        trie = CompletionTrieNode.build([(None, q) for q in titles])
        nb, gmks, gmksd, gmksd2, size = cmks(trie)
        gain, gain_dyn, gain_dyn2, ave_length = gain_dynamique_moyen_par_mot(titles, [
                                                                             1.0] * len(titles))
        fLOG("***", 1, nb, size, "*", gmks / size, gmksd / size, gmksd2 / size)
        fLOG("***", gain, gain_dyn, gain_dyn2, ave_length)
        self.assertEqual(nb, 494)


if __name__ == "__main__":
    unittest.main()
