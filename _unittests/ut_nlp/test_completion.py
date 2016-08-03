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
            st, ['[-::w=1]', '[#:a:w=4]', '[#:ab:w=2]', '[#:abc:w=3]', '[#:abcd:w=1]'])
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
        trie.update_stat_dynamic()
        for leave in trie.leaves():
            if leave.stat is None:
                raise Exception("None for {0}".format(leave))
            find = trie.find(leave.value)
            self.assertEqual(id(find), id(leave))
            assert hasattr(leave, "stat")
            assert hasattr(leave.stat, "mks0")
            assert hasattr(leave.stat, "mks")
            mk1 = trie.min_keystroke(leave.value)
            try:
                mk = trie.min_dynamic_keystroke(leave.value)
                mk2 = trie.min_dynamic_keystroke2(leave.value)
            except Exception as e:
                raise Exception(
                    "{0}-{1}-{2}-{3}".format(id(trie), id(leave), str(leave), leave.leave)) from e
            if mk[0] > mk1[0]:
                raise Exception("weird {0} > {1}".format(mk, mk1))
            if mk2[0] < mk[0]:
                raise Exception("weird {0} > {1}".format(mk, mk2))
            fLOG(leave.value, mk, "-", leave.stat.str_mks())
            self.assertEqual(
                mk, (leave.stat.mks0, leave.stat.mks0_, leave.stat.mksi_))
            text = leave.str_all_completions()
            assert text
            text = leave.str_all_completions(use_precompute=False)
            assert text

    def test_permutations(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        queries = ['actuellement', 'actualité', 'actu']
        weights = [1, 1, 0]
        for per in itertools.permutations(zip(queries, weights)):
            trie = CompletionTrieNode.build([(None, w) for w, p in per])
            trie.precompute_stat()
            trie.update_stat_dynamic()
            fLOG("----", per)
            for n in trie.leaves():
                fLOG("   ", n.value, n.stat.str_mks())
                assert n.stat.mks <= n.stat.mks0
                a, b, c = trie.min_dynamic_keystroke(n.value)
                self.assertEqual(a, n.stat.mks)
                a, b = trie.min_keystroke(n.value)
                if a != n.stat.mks0:
                    mes = [str(per)]
                    for n in trie.leaves():
                        mes.append("{0} - {1} || {2}".format(n.value,
                                                             n.stat.str_mks(), trie.min_keystroke(n.value)))
                    mes.append("---")
                    for n in trie:
                        mes.append("{0} || {1}".format(
                            n.value, n.stat.str_mks()))
                        for i, s in enumerate(n.stat.completions):
                            mes.append(
                                "  {0} - {1}:{2}".format(i, s[0], s[1].value))
                    raise Exception("difference\n{0}".format("\n".join(mes)))

    def test_normalize(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        this = os.path.abspath(os.path.dirname(__file__))
        this = os.path.join(this, "data", "wikititles.txt")
        with open(this, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip(" \r\n\t")
            cl = normalize_wiki_text(line)
            lo = remove_diacritics(cl).lower()
            fLOG(line, cl, lo)
            assert len(line) >= len(cl)
            assert len(line) >= len(lo)

    def test_load_titles(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        this = os.path.abspath(os.path.dirname(__file__))
        this = os.path.join(this, "data", "wikititles.txt")
        titles = sorted(enumerate_titles(this))
        res = {}
        dups = 0
        for w in titles:
            wc = remove_diacritics(w).lower()
            if wc not in res:
                res[wc] = w
            else:
                fLOG("duplicated key: '{0}', '{1}', key: '{2}'".format(
                    w, res[wc], wc))
                dups += 1
        fLOG("len(titles)=", len(res), "duplicated", dups)
        titles = list(sorted((None, k, v) for k, v in res.items()))
        self.assertEqual(titles[-1], (None, 'grand russe', 'Grand Russe'))
        self.assertEqual(
            titles[-2], (None, 'grand rue de pera', 'Grand Rue de Pera'))
        trie = CompletionTrieNode.build(titles)
        nodes = list(trie)
        exp_value = '[-:":w=0]'
        if str(nodes[1]) != exp_value:
            lines = "\n".join(str(_) for _ in nodes[:5])
            lines2 = "\n".join(str(_) for _ in titles[:5])
            info = ";".join(k for k, v in sorted(trie.children.items()))
            raise Exception("{0} != {1}\n{2}\nTITLES\n{3}\INFO\n{4}".format(
                str(nodes[1]), exp_value, lines, lines2, info))
        if str(nodes[-1]) != "[#:grand russe:w=354]":
            lines = "\n".join(str(_) for _ in nodes[-5:])
            lines2 = "\n".join(str(_) for _ in titles[-5:])
            raise Exception("{0} != {1}\n{2}\nTITLES\n{3}".format(
                str(nodes[-1]), "[#:grand russe:w=354]", lines, lines2))
        self.assertEqual(len(nodes), 3753)

        def cmks(trie):
            trie.precompute_stat()
            trie.update_stat_dynamic()
            gmks = 0.0
            gmksd = 0.0
            nb = 0
            size = 0
            for n in trie.leaves():
                gmks += len(n.value) - n.stat.mks0
                gmksd += len(n.value) - n.stat.mks
                size += len(n.value)
                nb += 1
            return nb, gmks, gmksd, size
        nb, gmks, gmksd, size = cmks(trie)
        fLOG(nb, size, gmks / nb, gmksd / nb, gmks / size, gmksd / size)
        assert gmks >= gmksd
        if gmksd == 0:
            i = 0
            for node in trie:
                fLOG(node.value, "--", node.stat.str_mks())
                if i > 20:
                    break
                i += 1
            assert False

        trie = CompletionTrieNode.build(titles)
        nb2, gmks2, gmksd2, size = cmks(trie)
        self.assertEqual(nb, nb2)
        self.assertEqual(gmks, gmks2)
        self.assertEqual(gmksd, gmksd2)
        assert gmksd > 0.62
        fLOG(nb2, gmks2 / nb2, gmksd2 / nb2)
        fLOG("-----")
        for i in range(1, 20):
            trie = CompletionTrieNode.build(titles[:i])
            nb, gmks, gmksd, size = cmks(trie)
            if i == 1:
                self.assertEqual(gmks, 30)
            fLOG(i, nb, size, gmks / nb, gmksd / nb,
                 gmks / size, gmksd / size, gmks)

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
            nb = 0
            size = 0
            for n in trie.leaves():
                gmks += len(n.value) - n.stat.mks0
                gmksd += len(n.value) - n.stat.mks
                size += len(n.value)
                nb += 1
            return nb, gmks, gmksd, size

        titles = [(None, '"contra el gang del chicharron"',
                   '"Contra el gang del chicharron')]
        trie = CompletionTrieNode.build(titles)
        nb, gmks, gmksd, size = cmks(trie)
        fLOG("***", 1, nb, size, gmks / nb, gmksd /
             nb, gmks / size, gmksd / size, gmks)
        self.assertEqual(gmks, 30)

        titles.append((None, '"la sequestree"', '"La séquestrée'))
        trie = CompletionTrieNode.build(titles)
        nb, gmks, gmksd, size = cmks(trie)
        fLOG("***", 2, nb, size, gmks / nb, gmksd /
             nb, gmks / size, gmksd / size, gmks)
        for n in trie.leaves():
            fLOG("***", n.value, n.stat.str_mks())
        self.assertEqual(gmks, 43)

    def test_duplicates(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        titles = ["abdcf", "abdcf"]
        try:
            fLOG(titles)
            trie = CompletionTrieNode.build(
                [(None, remove_diacritics(w).lower(), w) for w in titles])
            fLOG(trie)
            l = list(trie)
            assert len(l) == 6
            assert trie is not None
        except ValueError as e:
            fLOG(e)

    def test_completions(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        this = os.path.abspath(os.path.dirname(__file__))
        data = os.path.join(this, "data", "sample300.txt")
        with open(data, "r", encoding="utf-8") as f:
            lines = [_.strip(" \n\r\t") for _ in f.readlines()]

        trie = CompletionTrieNode.build([(None, q) for q in lines])
        trie.precompute_stat()
        trie.update_stat_dynamic()

        for q in lines:
            find = trie.find(q)
            assert find is not None
            sug = find.all_mks_completions()
            nb_ = [(a.value, len([s.value for _, s in b if s.value == q]))
                   for a, b in sug]
            nb = sum(_[1] for _ in nb_)
            if nb == 0:
                info = "nb={0} q='{1}'".format(nb, q)
                st = find.stat.str_mks()
                text = find.str_all_completions()
                text2 = find.str_all_completions(use_precompute=False)
                raise Exception(
                    "{4}\n---\nleave='{0}'\n{1}\n---\n{2}\n---\n{3}".format(find.value, st, text, text2, info))


if __name__ == "__main__":
    unittest.main()
