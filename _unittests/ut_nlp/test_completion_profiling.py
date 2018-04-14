# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)

https://dumps.wikimedia.org/frwiki/latest/frwiki-latest-all-titles.gz
https://dumps.wikimedia.org/frwiki/latest/frwiki-latest-all-titles-in-ns0.gz
"""

import sys
import os
import unittest
import cProfile
import pstats
import io


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
from pyquickhelper.pycode import get_temp_folder
from src.mlstatpy.nlp.completion import CompletionTrieNode


class TestCompletionProfiling(unittest.TestCase):

    def gain_dynamique_moyen_par_mot(self, queries, weights):
        per = list(zip(weights, queries))
        total = sum(weights) * 1.0
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

    def test_profiling(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        temp = get_temp_folder(__file__, "temp_profiling")
        data = os.path.join(temp, "..", "data", "sample1000.txt")
        with open(data, "r", encoding="utf-8") as f:
            lines = [_.strip(" \n\r\t") for _ in f.readlines()]

        def profile_exe():
            res = self.gain_dynamique_moyen_par_mot(lines, [1.0] * len(lines))
            return res

        def prof(n, show):
            pr = cProfile.Profile()
            pr.enable()
            profile_exe()
            pr.disable()
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats()
            rem = os.path.normpath(os.path.join(temp, "..", "..", ".."))
            res = s.getvalue().replace(rem, "")
            if show:
                fLOG(res)
            with open(os.path.join(temp, "profiling%d.txt" % n), "w") as f:
                f.write(res)
        prof(1, show=False)
        prof(2, show=True)


if __name__ == "__main__":
    unittest.main()
