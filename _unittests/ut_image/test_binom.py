import unittest
from mlstatpy.image.detection_segment import tabule_queue_binom


class TestQueueBinom(unittest.TestCase):
    def test_queue(self):
        b = tabule_queue_binom(2, 2)
        self.assertEqual(
            b,
            {
                (0, 1): 0.0,
                (1, 2): 0.0,
                (0, 0): 1.0,
                (2, 3): 0.0,
                (2, 0): 1.0,
                (1, 0): 1.0,
                (2, 2): 4.0,
                (1, 1): 2.0,
                (2, 1): 0.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
