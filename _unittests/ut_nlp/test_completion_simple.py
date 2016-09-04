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
from src.mlstatpy.nlp.completion_simple import CompletionSystem, CompletionElement


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
        diffs = trie.compare_with_trie(fLOG=fLOG)
        if diffs:
            res = [_[-1] for _ in diffs]
            raise Exception("\n".join(res))
        r = trie[2]
        assert r._info
        s = trie[2].str_all_completions()
        assert s
        assert isinstance(r._info._log_imp, list)
        for k, v in sorted(r._info._completions.items()):
            assert isinstance(v, list)
            if k != '' and len(v) > 2:
                raise Exception(v)
            assert v
            fLOG(k, v)
            for _ in v:
                fLOG("    ", _.value, ":", _.str_mks())
        assert "MKS=3 *=3 |'=3 *=3 |\"=3 *=3" in s
        assert trie.to_dict()

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

    def test_mks_consistency_port(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        titles = ["por", "por rouge", "por vert",
                  "por orange", "port", "port blanc", "port rouge"]
        trie = CompletionSystem(titles)
        diffs = trie.compare_with_trie()
        if diffs:
            res = [_[-1] for _ in diffs]
            raise Exception("\n".join(res))

        titles = ["po", "po rouge", "po vert", "po orange",
                  "port", "port blanc", "port rouge"]
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
        diffs = trie.compare_with_trie(fLOG=fLOG)
        if diffs:
            res = [_[-1] for _ in diffs]
            if len(res) > 3:
                res = res[:3]
            raise Exception("\n".join(res))
        assert len(trie) > 0

    def test_exception(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        try:
            e = CompletionElement(4, 5)
        except TypeError as e:
            assert "value must be str not '4'" in str(e)
        e = CompletionElement("4", 5)
        r = e.str_mks0()
        self.assertEqual(r, "-")
        r = e.str_mks()
        self.assertEqual(r, "-")

    def test_mks_consistency_bigger(self):

        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        def cmks(trie):
            diffs = trie.compare_with_trie(fLOG=fLOG)
            if diffs:
                if len(diffs) > 3:
                    diffs = diffs[:3]
                res = [_[-1] for _ in diffs]
                raise Exception("\n".join(res))

            gmks = 0.0
            gmksd = 0.0
            gmksd2 = 0.0
            nb = 0
            size = 0
            for n in trie:
                if n.mks2 < n.mks1 or (n.value == "baaaab" and n.mks1 != 4):
                    info = ""  # n.str_all_completions()
                    info2 = ""  # n.str_all_completions(use_precompute=True)
                    raise Exception("issue with query '{0}'\n{1}\n##########\n{2}\n############\n{3}".format(
                        n.value, n.str_mks(), info, info2))

                gmks += len(n.value) - n.mks0
                gmksd += len(n.value) - n.mks1
                gmksd2 += len(n.value) - n.mks2
                size += len(n.value)
                nb += 1
            return nb, gmks, gmksd, gmksd2, size

        def gain_dynamique_moyen_par_mot(queries, weights):
            per = list(zip(weights, queries))
            total = sum(w * len(q) for q, w in zip(queries, weights))
            trie = CompletionSystem([(None, q) for _, q in per])
            trie.compute_metrics()
            wks = [(w, p, len(w) - trie.find(w).mks0) for p, w in per]
            wks_dyn = [(w, p, len(w) - trie.find(w).mks1) for p, w in per]
            wks_dyn2 = [(w, p, len(w) - trie.find(w).mks2) for p, w in per]
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
        trie = CompletionSystem([(None, q) for q in titles])
        trie.compute_metrics(fLOG=fLOG, details=True)
        nb, gmks, gmksd, gmksd2, size = cmks(trie)
        gain, gain_dyn, gain_dyn2, ave_length = gain_dynamique_moyen_par_mot(titles, [
                                                                             1.0] * len(titles))
        fLOG("***", 1, nb, size, "*", gmks / size, gmksd / size, gmksd2 / size)
        fLOG("***", gain, gain_dyn, gain_dyn2, ave_length)
        self.assertEqual(nb, 494)

    def test_completions_bug(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        couleur = ["blanc", "vert", "orange", "rouge", "noir", "noire", "blanche"]
        key = "portes"
        mots = ["porch", "porch rouge", "porch vert", "porch orange", "pore", "pour"]
        mots.append(key)
        mots += [key + " " + c for c in couleur]
        ens = CompletionSystem(mots)        
        diffs = ens.compare_with_trie(fLOG=fLOG)
        if diffs:
            res = [_[-1] for _ in diffs]
            if len(res) > 3:
                res = res[:3]
            raise Exception("\n".join(res))
        assert len(ens) > 0
        m = ens.find("portes blanche")
        self.assertEqual(m.mks2, 7.8)

if __name__ == "__main__":
    unittest.main()
