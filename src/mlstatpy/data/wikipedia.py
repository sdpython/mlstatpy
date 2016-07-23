"""
@file
@brief Functions to retrieve data from Wikipedia
"""
import os
from pyquickhelper.loghelper import noLOG
from pyquickhelper.filehelper import get_url_content_timeout, ungzip_files
from .data_exceptions import DataException


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
    if unzip:
        names = ungzip_files(name, unzip=False)
        os.remove(name)
        if isinstance(names, list):
            if len(names) != 1:
                raise DataException(
                    "Expecting only one file, not '{0}'".format(names))
            return names[0]
        else:
            return names
    else:
        return name


def download_dump(country, name, folder, unzip=True, timeout=-1, fLOG=noLOG):
    """
    download wikipedia dumps from ``https://dumps.wikimedia.org/frwiki/latest/``

    @param      country     country
    @param      name        name of the stream to download
    @param      folder      where to download
    @param      unzip       unzip the file
    @param      timeout     timeout
    @param      fLOG        logging function
    """
    url = "https://dumps.wikimedia.org/{0}wiki/latest/{0}wiki-{1}".format(
        country, name)
    file = url.split("/")[-1]
    name = os.path.join(folder, file)
    get_url_content_timeout(url, timeout=timeout,
                            encoding=None, output=name, chunk=2**20, fLOG=fLOG)
    if unzip:
        names = ungzip_files(name, unzip=False)
        os.remove(name)
        if isinstance(names, list):
            if len(names) != 1:
                raise DataException(
                    "Expecting only one file, not '{0}'".format(names))
            return names[0]
        else:
            return names
    else:
        return name


def download_titles(country, folder, unzip=True, timeout=-1, fLOG=noLOG):
    """
    download wikipedia titles from ``https://dumps.wikimedia.org/frwiki/latest/latest-all-titles-in-ns0.gz``

    @param      country     country
    @param      folder      where to download
    @param      unzip       unzip the file
    @param      timeout     timeout
    @param      fLOG        logging function
    """
    return download_dump(country, "latest-all-titles-in-ns0.gz", folder, unzip=unzip, timeout=timeout, fLOG=fLOG)
