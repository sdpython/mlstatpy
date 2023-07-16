"""
@brief      test log(time=2s)

"""
import os
import unittest
import copy
import warnings
from pyquickhelper.pycode import get_temp_folder, is_travis_or_appveyor, ExtTestCase
from mlstatpy.graph.graph_distance import GraphDistance
from mlstatpy.graph.graphviz_helper import draw_graph_graphviz


class TestGraphDistance(ExtTestCase):
    def test_graph_load(self):
        this = os.path.abspath(os.path.dirname(__file__))
        graph = os.path.join(this, "data", "graph.gv")
        g = GraphDistance.load_from_file(graph, False)
        paths = list(g.enumerate_all_paths(True))
        self.assertTrue(len(paths) > 0)

    def test_image_video_kohonen(self):
        temp = get_temp_folder(__file__, "temp_graph_distance")

        graph1 = [
            ("a", "b"),
            ("b", "c"),
            ("b", "d"),
            ("d", "e"),
            ("e", "f"),
            ("b", "f"),
            ("b", "g"),
            ("f", "g"),
            ("a", "g"),
            ("a", "g"),
            ("c", "d"),
            ("d", "g"),
            ("d", "h"),
            ("aa", "h"),
            ("aa", "c"),
            ("f", "h"),
        ]
        graph2 = copy.deepcopy(graph1) + [
            ("h", "m"),
            ("m", "l"),
            ("l", "C"),
            ("C", "r"),
            ("a", "k"),
            ("k", "l"),
            ("k", "C"),
        ]

        graph1 = GraphDistance(graph1)
        graph2 = GraphDistance(graph2)

        graph2["C"].label = "c"
        store = {}
        if len(list(graph1.enumerate_all_paths(True))) != 17:
            raise AssertionError("expecting 17 here")
        if len(list(graph2.enumerate_all_paths(True))) != 19:
            raise AssertionError("expecting 19 here")

        distance, graph = graph1.distance_matching_graphs_paths(
            graph2, use_min=False, store=store
        )

        if graph["h"].Label != "h":
            raise AssertionError("we expect this node to be merged in the process")

        if distance is None:
            raise AssertionError("expecting something different from None")

        outfile1 = os.path.join(temp, "unittest_GraphDistance4_sub1.png")
        outfile2 = os.path.join(temp, "unittest_GraphDistance4_sub2.png")
        outfilef = os.path.join(temp, "unittest_GraphDistance4_subf.png")

        if is_travis_or_appveyor() == "travis":
            warnings.warn("graphviz is not available")
            return

        vertices, edges = graph1.draw_vertices_edges()
        self.assertNotEmpty(vertices)
        self.assertNotEmpty(edges)
        draw_graph_graphviz(vertices, edges, outfile1)

        vertices, edges = graph2.draw_vertices_edges()
        self.assertNotEmpty(vertices)
        self.assertNotEmpty(edges)
        draw_graph_graphviz(vertices, edges, outfile2)
        self.assertTrue(os.path.exists(outfile2))

        vertices, edges = graph.draw_vertices_edges()
        self.assertNotEmpty(vertices)
        self.assertNotEmpty(edges)
        draw_graph_graphviz(vertices, edges, outfilef)
        self.assertTrue(os.path.exists(outfilef))

    def test_unittest_GraphDistance2(self):
        graph1 = [
            ("a", "b"),
            ("b", "c"),
            ("b", "X"),
            ("X", "c"),
            ("c", "d"),
            ("d", "e"),
            ("0", "b"),
        ]
        graph2 = [
            ("a", "b"),
            ("b", "c"),
            ("b", "X"),
            ("X", "c"),
            ("c", "t"),
            ("t", "d"),
            ("d", "e"),
            ("d", "g"),
        ]
        graph1 = GraphDistance(graph1)
        graph2 = GraphDistance(graph2)
        store = {}
        res, out, err = self.capture(
            lambda: graph1.distance_matching_graphs_paths(
                graph2, use_min=False, store=store, verbose=True
            )
        )
        self.assertIn("[distance_matching_graphs_paths]", out)
        self.assertIn("#", err)
        distance, graph = res
        if distance is None:
            raise TypeError("expecting something different from None")
        allPaths = list(graph.enumerate_all_paths(True))
        if len(allPaths) == 0:
            raise ValueError("the number of paths should not be null")
        if distance == 0:
            raise ValueError("expecting a distance > 0")
        vertices, edges = graph.draw_vertices_edges()
        self.assertNotEmpty(vertices)
        self.assertNotEmpty(edges)
        # GV.drawGraphEdgesVertices (vertices,edges, "unittest_GraphDistance2.png")
        node = graph.vertices["X"]
        if None in node.pair:
            raise RuntimeError("unexpected, this node should be part of the common set")

        vertices, edges = graph1.draw_vertices_edges()
        self.assertNotEmpty(vertices)
        # GV.drawGraphEdgesVertices (vertices,edges, "unittest_GraphDistance2_sub1.png")
        vertices, edges = graph2.draw_vertices_edges()
        self.assertNotEmpty(vertices)
        # GV.drawGraphEdgesVertices (vertices,edges, "unittest_GraphDistance2_sub2.png")

    def test_unittest_common_paths(self):
        graph1 = [
            ("a", "b"),
            ("b", "c"),
            ("b", "X"),
            ("X", "c"),
            ("c", "d"),
            ("d", "e"),
            ("0", "b"),
        ]
        graph2 = graph1
        graph1 = GraphDistance(graph1)
        graph2 = GraphDistance(graph2)
        common12 = graph1.common_paths(graph2)
        common21 = graph2.common_paths(graph1)
        s1 = str(common12)
        s2 = repr(common12)
        self.assertIn("c-c -> d-d []", s1)
        self.assertIn("[Edge('0-0', 'b-b', '', 1.0)", s2)
        self.assertIn("{'0-0': Vertex('0-0', '0', 1.0)", s2)
        self.assertEqual(len(graph1.vertices), len(common12.vertices))
        self.assertEqual(len(graph1.vertices), len(common21.vertices))
        self.assertEqual(len(graph1.edges), len(common21.edges))


if __name__ == "__main__":
    unittest.main()
