# -*- coding: utf-8 -*-
"""
@brief      test log(time=60s)
"""
import unittest
from datetime import datetime
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import get_temp_folder, ExtTestCase
from mlstatpy.data.wikipedia import download_pageviews


class TestLONGWikipediaPageViews(ExtTestCase):
    def test_wikipedia_page_views(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")

        temp = get_temp_folder(__file__, "temp_wikipedia_pageviews")
        name = download_pageviews(datetime(2016, 5, 6, 10), folder=temp, fLOG=fLOG)
        self.assertLesser(len(name), 2000)
        self.assertExists(name)


if __name__ == "__main__":
    unittest.main()
