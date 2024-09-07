import unittest
from mlstatpy.nlp.completion_simple import CompletionSystem


class TestCompletionSimpleOptimisation(unittest.TestCase):
    def test_build_trie_simple(self):
        comp = [(1, "a"), (2, "ab"), (3, "abc"), (4, "abcd"), (5, "bc")]
        cset = CompletionSystem(comp)
        cset.compute_metrics()
        queries = [(q, w) for w, q in comp]
        res = cset.test_metric(queries)
        self.assertEqual(res["mks1"], res["sum_wlen"])
        self.assertEqual(res["hist"]["l"], {1: 1, 2: 7, 3: 3, 4: 4})

        comp = [q for w, q in comp]
        comp.reverse()
        cset = CompletionSystem(comp)
        cset.compute_metrics()
        queries = [(q, 1) for q in comp]
        for _el, found in cset.enumerate_test_metric(queries):
            # print(el, found)
            assert found is not None
        res = cset.test_metric(queries)
        # for k, v in sorted(res.items()): print(k, "=", v)
        assert res["mks1"] < res["sum_wlen"]
        self.assertEqual(res["n"], 5)
        self.assertEqual(res["hist"]["l"], {1: 1, 2: 2, 3: 1, 4: 1})

        # one suggestion in the completion set
        comp = ["a", "abc", "bc"]
        cset = CompletionSystem(comp)
        cset.compute_metrics()
        queries = [(q, 1) for q in comp] + [("abcd", 1)]
        for el, found in cset.enumerate_test_metric(queries):
            # if found is None:
            # print(el.str_mks(), "*", el.value)
            # else:
            # print(el.str_mks(), "*", el.value, "*", found, found.str_mks())
            # print(el.weight)
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
        # for k, v in sorted(res.items()):
        # print(k, "=", v)
        assert res["mks1"] < res["sum_wlen"]
        self.assertEqual(res["n"], 4)
        self.assertEqual(res["hist"]["l"], {1: 1, 2: 1, 3: 1, 4: 1})


if __name__ == "__main__":
    unittest.main()
