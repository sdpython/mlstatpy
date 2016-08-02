#-*- coding: utf-8 -*-
"""
@brief      test log(time=24s)
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


class TestLONGCompletion(unittest.TestCase):

    def test_build_dynamic_trie_mks_min(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        data = os.path.join(os.path.abspath(
            os.path.dirname(__file__)), "data", "sample20000.txt")
        with open(data, "r", encoding="utf-8") as f:
            lines = [_.strip("\n\r\t ") for _ in f.readlines()]
        queries = [(None, _) for _ in lines]
        fLOG("build trie")
        trie = CompletionTrieNode.build(queries)
        fLOG(len(queries), len(set(_[1] for _ in queries)),
             len(list(trie.leaves())), len(set(trie.leaves())))
        assert "Cannes 2005" in set(_[1] for _ in queries)
        assert "Cannes 2005" in set(_.value for _ in trie.leaves())
        fLOG("precompute")
        trie.precompute_stat()
        fLOG("update")
        trie.update_stat_dynamic()
        fLOG("loop")
        for i, q in enumerate(queries):
            if i % 1000 == 0:
                fLOG(i)
            leave = trie.find(q[1])
            if leave.stat is None:
                raise Exception("None for {0}".format(leave))
            assert hasattr(leave, "stat")
            assert hasattr(leave.stat, "mks0")
            assert hasattr(leave.stat, "mks")

            sug = leave.all_mks_completions()
            nb_ = [(a.value, len([s.value for _, s in b if s.value == q[1]]))
                   for a, b in sug]
            nbf_ = [(a.value, len(b)) for a, b in sug]
            nb = sum(_[1] for _ in nb_)
            mnb = max(_[1] for _ in nbf_)
            if nb == 0 and len(q[1]) > 10:
                info = "nb={0} mnb={2} q='{1}'".format(nb, q[1], mnb)
                st = leave.stat.str_mks()
                text = leave.str_all_completions()
                text2 = leave.str_all_completions(use_precompute=False)
                raise Exception(
                    "{4}\n---\nleave='{0}'\n{1}\n---\n{2}\n---\n{3}".format(leave.value, st, text, text2, info))

            mk1 = trie.min_keystroke0(leave.value)
            try:
                mk = trie.min_dynamic_keystroke(leave.value)
                mk2 = trie.min_dynamic_keystroke2(leave.value)
            except Exception as e:
                raise Exception(
                    "{0}-{1}-{2}-{3}".format(id(trie), id(leave), str(leave), leave.leave)) from e

            if mk[0] > mk1[0]:
                st = leave.stat.str_mks()
                text = leave.str_all_completions()
                text2 = leave.str_all_completions(use_precompute=False)
                raise Exception("weird {0} > {1} -- leave='{2}'\n{3}\n---\n{4}\n---\n{5}".format(
                    mk, mk1, leave.value, st, text, text2))
            if mk2[0] < mk[0]:
                st = leave.stat.str_mks()
                text = leave.str_all_completions()
                text2 = leave.str_all_completions(use_precompute=False)
                raise Exception("weird {0} > {1} -- leave='{2}'\n{3}\n---\n{4}\n---\n{5}".format(
                    mk, mk2, leave.value, st, text, text2))


if __name__ == "__main__":
    unittest.main()
