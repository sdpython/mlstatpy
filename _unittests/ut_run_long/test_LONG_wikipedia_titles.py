# -*- coding: utf-8 -*-
"""
@brief      test log(time=60s)
"""
import unittest
from pyquickhelper.pycode import get_temp_folder, ExtTestCase
from mlstatpy.data.wikipedia import download_titles


class TestLONGWikipediaPageCount(ExtTestCase):
    def test_wikipedia_page_count(self):
        temp = get_temp_folder(__file__, "temp_wikipedia_title")
        name = download_titles("fr", folder=temp)
        self.assertLesser(len(name), 2000)
        self.assertExists(name)


if __name__ == "__main__":
    unittest.main()
