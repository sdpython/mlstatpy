"""
@file
@brief Functions to retrieve data from Wikipedia
"""
import os
from pyquickhelper.loghelper import noLOG
from pyquickhelper.filehelper import get_url_content_timeout


def download_pagecount(dt, folder, unzip=True, timeout=-1, fLOG=noLOG):
    """
    download wikipedia pagacount for a precise date (up to the hours),
    the url follows the pattern::

        https://dumps.wikimedia.org/other/pagecounts-raw/%Y/%Y-%m/pagecounts-%Y%m%d-%H0000.gz

    @param      dt          datetime
    @param      folder      where to download
    @param      unzip       unzip the file
    @param      timeout     timeout
    @param      fLOG        logging function
    @return                 filename

    More information on page `pagecounts-raw <https://dumps.wikimedia.org/other/pagecounts-raw/>`_.
    """
    url = "https://dumps.wikimedia.org/other/pagecounts-raw/%Y/%Y-%m/pagecounts-%Y%m%d-%H0000.gz"
    url = dt.strftime(url)
    file = url.split("/")[-1]
    name = os.path.join(folder, file)
    get_url_content_timeout(url, timeout=timeout,
                            encoding=None, output=name, chunk=2**20, fLOG=fLOG)
    return name
