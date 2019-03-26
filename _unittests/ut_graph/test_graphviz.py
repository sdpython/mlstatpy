"""
@brief      test log(time=2s)

"""
import os
import unittest
from pyquickhelper.pycode import get_temp_folder, skipif_travis, skipif_appveyor
from mlstatpy.graph.graphviz_helper import draw_graph_graphviz


class TestGraphviz(unittest.TestCase):

    @skipif_appveyor("no graphviz")
    @skipif_travis("no graphviz")
    def test_make_video(self):
        temp = get_temp_folder(__file__, "temp_graphviz")
        fout = os.path.join(temp, "image.png")

        draw_graph_graphviz([(1, "eee", "red")],
                            [(1, 2, "blue"), (3, 4), (1, 3)], fout)

        self.assertTrue(os.path.exists(fout))
        self.assertTrue(os.path.exists(fout + ".gv"))


if __name__ == "__main__":
    unittest.main()
