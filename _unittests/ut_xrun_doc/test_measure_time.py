import unittest
from math import cos
from teachpyx.ext_test_case import ExtTestCase, measure_time


class TestMeasureTime(ExtTestCase):
    def test_measure_time(self):
        res = measure_time(lambda: cos(5))
        self.assertIsInstance(res, dict)
        self.assertIn("average", res)


if __name__ == "__main__":
    unittest.main(verbosity=2)
