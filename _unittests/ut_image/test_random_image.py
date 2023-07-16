# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import os
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from mlstatpy.image.detection_segment.random_image import (
    random_noise_image,
    random_segment_image,
)
from mlstatpy.image.detection_segment import convert_array2PIL, convert_PIL2array
from mlstatpy.image.detection_segment.detection_segment import detect_segments


class TestRandomImage(ExtTestCase):
    def test_random_noise_image(self):
        img = random_noise_image((100, 100), 0.1)
        total = img.sum()
        self.assertGreater(total, 0)
        self.assertLesser(total, 3000)

    def test_random_segment_image(self):
        img = random_noise_image((12, 10), 0.0)
        seg = random_segment_image(img, lmin=0.5, density=2.0)
        total = img.sum()
        self.assertGreater(total, 0)
        self.assertLesser(total, 3000)
        self.assertIsInstance(seg, dict)

        fimg = img.astype(numpy.float32)
        img255 = (-fimg + 1) * 255
        timg255 = img255.astype(numpy.uint8)
        pil = convert_array2PIL(timg255)
        img2 = convert_PIL2array(pil)
        temp = get_temp_folder(__file__, "temp_random_segment_image")
        outfile = os.path.join(temp, "img.png")
        pil.save(outfile)
        self.assertEqual(timg255, img2)

        pil2 = convert_array2PIL(img, mode="binary")
        img3 = convert_PIL2array(pil2)
        self.assertEqual(timg255, img3)

        for _ in range(0, 100):
            seg = random_segment_image(img, lmin=0.5, density=2.0)
            self.assertGreater(seg["x1"], 0)
            self.assertGreater(seg["y1"], 0)
            self.assertGreater(seg["x2"], 0)
            self.assertGreater(seg["y2"], 0)

    def test_segment_random_image(self):
        img = random_noise_image((100, 100))
        random_segment_image(img, density=3.0, lmin=0.3)
        random_segment_image(img, density=5.0, lmin=0.3)
        random_segment_image(img, density=5.0, lmin=0.3)
        seg = detect_segments(img, seuil_nfa=10, seuil_norme=1, verbose=0)
        # self.assertNotEmpty(seg)
        self.assertTrue(seg is not None)


if __name__ == "__main__":
    unittest.main()
