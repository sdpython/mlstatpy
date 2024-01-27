"""
@brief      test log(time=2s)

"""

import os
import unittest
from pyquickhelper.pycode import get_temp_folder
from mlstatpy.graph.graphviz_helper import draw_graph_graphviz


class TestGraphviz(unittest.TestCase):
    def test_draw_graph_graphviz(self):
        temp = get_temp_folder(__file__, "temp_graphviz")
        fout = os.path.join(temp, "image.png")

        try:
            draw_graph_graphviz(
                [(1, "eee", "red")], [(1, 2, "blue"), (3, 4), (1, 3)], fout
            )
        except FileNotFoundError as e:
            if "No such file or directory: 'dot'" in str(e):
                return
            raise e

        self.assertTrue(os.path.exists(fout))
        self.assertTrue(os.path.exists(fout + ".gv"))

    def test_draw_graph_graphviz_no_image(self):
        try:
            res = draw_graph_graphviz(
                [(1, "eee", "red")], [(1, 2, "blue"), (3, 4), (1, 3)], image=None
            )
        except FileNotFoundError as e:
            if "No such file or directory: 'dot'" in str(e):
                return
            raise e
        self.assertIn('[label="eee"', res)


if __name__ == "__main__":
    unittest.main()
