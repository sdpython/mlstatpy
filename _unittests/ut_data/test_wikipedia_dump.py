# -*- coding: utf-8 -*-
"""
@brief      test log(time=60s)
"""
import unittest
from pyquickhelper.pycode import get_temp_folder, ExtTestCase
from mlstatpy.data.wikipedia import download_dump


class TestWikipediaDump(ExtTestCase):
    def test_wikipedia_dump(self):
        temp = get_temp_folder(__file__, "temp_wikipedia_abstract_gz")
        name = download_dump(
            "fr", "latest-abstract.xml.gz-rss.xml", folder=temp, unzip=False
        )
        # print(name)
        self.assertTrue(name is not None)
        self.assertExists(name)

    def test_wikipedia_dump_zipped(self):
        temp = get_temp_folder(__file__, "temp_wikipedia_dump_gz")
        name = download_dump("fr", "latest-site_stats.sql.gz", folder=temp, unzip=True)
        # print(name)
        self.assertTrue(name is not None)
        self.assertExists(name)
        self.assertTrue(not name.endswith("gz"))


if __name__ == "__main__":
    unittest.main()
