# -*- coding: utf-8 -*-
"""
@brief      test log(time=10s)
"""
import os
import unittest
import math
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from mlstatpy.image.detection_segment.geometrie import Point
from mlstatpy.image.detection_segment.detection_segment_segangle import SegmentBord
from mlstatpy.image.detection_segment.detection_segment import detect_segments, plot_segments
from mlstatpy.image.detection_segment.detection_segment import _calcule_gradient, plot_gradient
from mlstatpy import __file__ as rootfile


class TestSegments(ExtTestCase):

    visual = False

    def test_segment_bord(self):
        s = SegmentBord(Point(3, 4))
        n = True
        res = []
        while n:
            res.append(s.copy())
            n = s.next()  # pylint: disable=E1102
        self.assertEqual(len(res), 279)
        self.assertEqual(res[-1].a, Point(0, 3))
        self.assertEqual(res[-1].b, Point(7, 2))
        self.assertEqual(res[-2].a, Point(0, 0))
        self.assertEqual(res[-2].b, Point(6, 0))

    def test_segment_bord2(self):
        """
        Ceci n'est execute que si ce fichier est le fichier principal,
        permet de verifier que tous les segments sont bien parcourus."""
        xx, yy = 163, 123

        if TestSegments.visual and __name__ == "__main__":

            def attendre_clic(screen):
                """attend la pression d'un clic de souris
                avant de continuer l'execution du programme,
                methode pour pygame"""
                pygame.display.flip()
                reste = True
                while reste:
                    for event in pygame.event.get():
                        if event.type == pygame.MOUSEBUTTONUP:
                            reste = False
                            break

            import pygame  # pylint: disable=C0415
            pygame.init()
            screen = pygame.display.set_mode((xx * 4, yy * 4))
            screen.fill((255, 255, 255))
            pygame.display.flip()

            for i in range(1, 4):
                pygame.draw.line(screen, (255, 0, 0),
                                 (0, i * yy), (xx * 4, i * yy))
                pygame.draw.line(screen, (255, 0, 0),
                                 (xx * i, 0), (xx * i, 4 * yy))

        s = SegmentBord(Point(xx, yy), math.pi / 6)
        s.premier()

        i = 0
        n = True
        angle = 0
        x, y = 0, 0
        couleur = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
                   (255, 0, 255), (0, 0, 0), (128, 128, 128)]
        segs = []
        c = 0
        while n:
            if TestSegments.visual and __name__ == "__main__" and i % 100 == 0:
                print(f"i={i} s={s}")

            x = s.bord1
            y = s.calcul_bord2()
            a = (int(s.a.x) + x * xx, int(s.a.y) + y * yy)
            b = (int(s.b.x) + x * xx, int(s.b.y) + y * yy)

            if TestSegments.visual and __name__ == "__main__":
                pygame.draw.line(screen, couleur[c % len(couleur)], a, b)
                pygame.display.flip()

            n = s.next()  # pylint: disable=E1102
            if angle != s.angle:
                if TestSegments.visual and __name__ == "__main__":
                    print("changement angle = ", angle,
                          " --> ", s.angle, "   clic ", s)
                    pygame.draw.line(screen, couleur[c % len(couleur)], a, b)
                    pygame.display.flip()
                    # attendre_clic(screen)
                c += 1
            angle = s.angle
            segs.append(s.copy())
            i += 1

        if TestSegments.visual and __name__ == "__main__":
            pygame.display.flip()
            attendre_clic(screen)

        self.assertEqual(len(segs), 2852)
        seg = segs[-1]
        self.assertEqual(seg.a.x, 0)
        self.assertEqual(seg.a.y, 122)
        self.assertEqual(seg.b.x, 286)
        self.assertEqual(seg.b.y, 122)

    def test_gradient_profile(self):
        img = os.path.join(os.path.dirname(__file__),
                           "data", "eglise_zoom2.jpg")
        rootrem = os.path.normpath(os.path.abspath(
            os.path.join(os.path.dirname(rootfile), '..')))
        _, res = self.profile(lambda: _calcule_gradient(  # pylint: disable=W0632
            img, color=0), rootrem=rootrem)
        short = "\n".join(res.split('\n')[:15])
        self.assertIn("_calcule_gradient", short)

    def test_gradient(self):
        temp = get_temp_folder(__file__, "temp_segment_gradient")
        img = os.path.join(temp, "..", "data", "eglise_zoom2.jpg")
        grad = _calcule_gradient(img, color=0)
        self.assertEqual(grad.shape, (308, 408, 2))
        for d in [-2, -1, 0, 1, 2]:
            imgrad = plot_gradient(img, grad, direction=d)
            grfile = os.path.join(temp, "gradient-%d.png" % d)
            imgrad.save(grfile)
            self.assertExists(grfile)

        with open(os.path.join(temp, "..", "data", "gradient--2.png"), 'rb') as f:
            c1 = f.read()
        with open(os.path.join(temp, "..", "data", "gradient--2b.png"), 'rb') as f:
            c1b = f.read()
        with open(os.path.join(temp, "gradient--2.png"), 'rb') as f:
            c2 = f.read()
        self.assertIn(c2, (c1, c1b))

    def test_segment_detection_profile(self):
        img = os.path.join(os.path.dirname(__file__),
                           "data", "eglise_zoom2.jpg")
        rootrem = os.path.normpath(os.path.abspath(
            os.path.join(os.path.dirname(rootfile), '..')))
        _, res = self.profile(lambda: detect_segments(  # pylint: disable=W0632
            img, stop=100), rootrem=rootrem)
        short = "\n".join(res.split('\n')[:25])
        if __name__ == "__main__":
            print(short)
        self.assertIn("detect_segments", short)

    def test_segment_detection(self):
        temp = get_temp_folder(__file__, "temp_segment_detection")
        img = os.path.join(temp, "..", "data", "eglise_zoom2.jpg")
        outfile = os.path.join(temp, "seg.png")
        seg = detect_segments(img, stop=100)
        plot_segments(img, seg, outfile=outfile)
        self.assertIsInstance(seg, list)
        self.assertEqual(len(seg), 107)
        seg.sort()
        self.assertGreater(len(seg), 0)


if __name__ == "__main__":
    unittest.main()
