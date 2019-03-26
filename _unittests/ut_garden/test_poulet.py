# -*- coding: utf-8 -*-
"""
@brief      test log(time=4s)
"""
import unittest
from mlstatpy.garden.poulet import maximum, find_maximum, histogramme_poisson_melange, proba_poisson_melange


class TestPoulet(unittest.TestCase):

    def test_poulet1(self):
        res = maximum(2, 5, 1, 80)
        m = find_maximum(res)
        self.assertEqual(m, (86, 228.50205712688214))
        self.assertEqual(
            res[:3], [(0, 0.0), (1, 2.9999999999999942), (2, 5.9999999999999885)])

    def test_poulet2(self):
        h = histogramme_poisson_melange([48, 10, 4], [1, 2, 3])
        self.assertTrue(max(h) > 0.01)
        self.assertEqual(h[:4], [0.0, 0.0, 0.0, 0.0])

    def test_poulet3(self):
        h = proba_poisson_melange([48, 10, 4], [1, 2, 3], 20)
        self.assertEqual(h, 0)
        h = proba_poisson_melange([48, 10, 4], [1, 2, 3], 40)
        self.assertTrue(h < 0.1)


if __name__ == "__main__":
    unittest.main()
