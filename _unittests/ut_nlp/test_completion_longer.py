# -*- coding: utf-8 -*-
"""
@brief      test log(time=16s)
"""
import os
import unittest
from pyquickhelper.loghelper import fLOG
from mlstatpy.nlp.completion import CompletionTrieNode


class TestCompletionLonger(unittest.TestCase):

    def test_check_bug_about_mergeing_completions(self):
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
        fLOG("bug precompute")
        trie.precompute_stat()
        fLOG("bug checking")
        find = trie.find('Cann')
        sug = find.stat.completions
        self.assertEqual(len(sug), 2)
        leave = trie.find('Cannes 2005')

        sugg = leave.all_mks_completions()
        assert len(sugg) > 0
        verif = 0
        for p, sug in sugg:
            if p.value.startswith("Cannes"):
                for s in sug:
                    if s[1].value == "Cannes 2005":
                        verif += 1
        if verif == 0:
            raise AssertionError(leave.str_all_completions(use_precompute=True))

        sugg = leave.all_completions()
        assert len(sugg) > 0
        verif = 0
        for p, sug in sugg:
            if p.value.startswith("Cannes"):
                for s in sug:
                    if s == "Cannes 2005":
                        verif += 1
        if verif == 0:
            raise AssertionError(leave.str_all_completions(use_precompute=False))


if __name__ == "__main__":
    unittest.main()
