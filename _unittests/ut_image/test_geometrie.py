# -*- coding: utf-8 -*-
"""
@brief      test log(time=38s)
"""
import unittest
import math
from mlstatpy.image.detection_segment import Point, Segment


class TestGeometrie(unittest.TestCase):

    def test_point(self):
        p = Point(2, 2)
        pp = Point(3, 5)
        pp += p
        pp += p
        self.assertEqual(p, Point(2, 2))
        self.assertEqual(pp, Point(7, 9))
        pp -= p
        self.assertEqual(pp, Point(5, 7))
        pp.scalairek(0.5)
        self.assertEqual(pp, Point(2.5, 3.5))
        ar = pp.arrondi()
        self.assertEqual(ar, Point(3, 4))
        sc = ar.scalaire(ar)
        no = ar.norme()**2
        self.assertEqual(sc, no)
        a = Point(1, 1).angle()
        b = Point(-1, 1).angle()
        d = b - a
        dd = d - math.pi / 2
        assert abs(dd) < 1e-5
        seg = Segment(Point(0, 0), p)
        self.assertEqual(str(seg), "[(0,0),(2,2)]")
        n = seg.directeur().norme()
        assert abs(n - 1) < 1e-8
        d = seg.directeur()
        n = seg.normal()
        s = d.scalaire(n)
        assert abs(s) < 1e-8


if __name__ == "__main__":
    unittest.main()
