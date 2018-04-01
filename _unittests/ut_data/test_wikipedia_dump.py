#-*- coding: utf-8 -*-
"""
@brief      test log(time=60s)
"""

import sys
import os
import unittest


try:
    import pyquickhelper as skip_
except ImportError:
    path = os.path.normpath(
        os.path.abspath(
            os.path.join(
                os.path.split(__file__)[0],
                "..",
                "..",
                "..",
                "pyquickhelper",
                "src")))
    if path not in sys.path:
        sys.path.append(path)
    import pyquickhelper as skip_


try:
    import src
except ImportError:
    path = os.path.normpath(
        os.path.abspath(
            os.path.join(
                os.path.split(__file__)[0],
                "..",
                "..")))
    if path not in sys.path:
        sys.path.append(path)
    import src

from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import get_temp_folder
from src.mlstatpy.data.wikipedia import download_dump


class TestWikipediaDump(unittest.TestCase):

    def test_wikipedia_dump(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        temp = get_temp_folder(__file__, "temp_wikipedia_dump")
        name = download_dump("fr", "latest-abstract.xml.gz-rss.xml",
                             folder=temp, fLOG=fLOG, unzip=False)
        fLOG(name)
        assert name is not None
        assert os.path.exists(name)

    def test_wikipedia_dump_zipped(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        temp = get_temp_folder(__file__, "temp_wikipedia_dump_gz")
        name = download_dump("fr", "latest-site_stats.sql.gz",
                             folder=temp, fLOG=fLOG, unzip=True)
        fLOG(name)
        assert name is not None
        assert os.path.exists(name)
        assert not name.endswith("gz")


if __name__ == "__main__":
    unittest.main()
