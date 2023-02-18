# -*- coding: utf-8 -*-
"""
@brief      test log(time=33s)
"""
import os
import unittest
from pyquickhelper.loghelper import fLOG, CustomLog
from pyquickhelper.pycode import get_temp_folder
from mlstatpy.nlp.completion import CompletionTrieNode


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
        temp = get_temp_folder(__file__, "temp_build_dynamic_trie_mks_min")
        clog = CustomLog(temp)
        clog("build trie")
        trie = CompletionTrieNode.build(queries)
        fLOG(len(queries), len(set(_[1] for _ in queries)),
             len(list(trie.leaves())), len(set(trie.leaves())))

        self.assertTrue("Cannes 2005" in set(_[1] for _ in queries))
        self.assertTrue("Cannes 2005" in set(_.value for _ in trie.leaves()))

        clog("precompute")
        trie.precompute_stat()
        clog("update")
        trie.update_stat_dynamic()
        clog("loop")
        fLOG("loop")
        for i, q in enumerate(queries):
            if i % 1000 == 0:
                clog(i)
                fLOG(i)
            leave = trie.find(q[1])
            if leave.stat is None:
                raise AssertionError(f"None for {leave}")

            self.assertTrue(hasattr(leave, "stat"))
            self.assertTrue(hasattr(leave.stat, "mks0"))
            self.assertTrue(hasattr(leave.stat, "mks1"))

            sug = leave.all_mks_completions()
            nb_ = [(a.value, len([s.value for _, s in b if s.value == q[1]]))
                   for a, b in sug]
            nbf_ = [(a.value, len(b)) for a, b in sug]
            nb = sum(_[1] for _ in nb_)
            mnb = max(_[1] for _ in nbf_)
            if nb == 0 and len(q[1]) > 10:
                info = f"nb={nb} mnb={mnb} q='{q[1]}'"
                st = leave.stat.str_mks()
                text = leave.str_all_completions()
                text2 = leave.str_all_completions(use_precompute=False)
                raise AssertionError(
                    f"{info}\n---\nleave='{leave.value}'\n{st}\n---\n{text}\n---\n{text2}")

            mk1 = trie.min_keystroke0(leave.value)
            try:
                mk = trie.min_dynamic_keystroke(leave.value)
                mk2 = trie.min_dynamic_keystroke2(leave.value)
            except Exception as e:
                raise RuntimeError(
                    f"{id(trie)}-{id(leave)}-{str(leave)}-{leave.leave}") from e

            if mk[0] > mk1[0]:
                st = leave.stat.str_mks()
                text = leave.str_all_completions()
                text2 = leave.str_all_completions(use_precompute=False)
                raise RuntimeError(
                    "weird {0} > {1} -- leave='{2}'\n{3}\n---\n"
                    "{4}\n---\n{5}".format(
                        mk, mk1, leave.value, st, text, text2))
            if mk2[0] < mk[0]:
                st = leave.stat.str_mks()
                text = leave.str_all_completions()
                text2 = leave.str_all_completions(use_precompute=False)
                raise RuntimeError(
                    "weird {0} > {1} -- leave='{2}'\n{3}\n---\n{4}\n---\n{5}".format(
                        mk, mk2, leave.value, st, text, text2))
        clog("end")
        fLOG("end")


if __name__ == "__main__":
    unittest.main()
