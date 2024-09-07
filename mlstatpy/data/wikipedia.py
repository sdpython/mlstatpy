import os
from mlstatpy.ext_test_case import get_url_content_timeout, ungzip_files
from .data_exceptions import DataException


def download_pageviews(dt, folder=".", unzip=True, timeout=-1, overwrite=False):
    """
    Downloads wikipedia pagacount for a precise date (up to the hours),
    the url follows the pattern::

        https://dumps.wikimedia.org/other/pageviews/%Y/%Y-%m/pagecounts-%Y%m%d-%H0000.gz

    :param dt: datetime
    :param folder: where to download
    :param unzip: unzip the file
    :param timeout: timeout
    :param overwrite: overwrite
    :return: filename

    More information on page
    `pageviews <https://dumps.wikimedia.org/other/pageviews/>`_.
    """
    url = "https://dumps.wikimedia.org/other/pageviews/%Y/%Y-%m/pageviews-%Y%m%d-%H0000.gz"
    url = dt.strftime(url)
    file = url.split("/")[-1]
    name = os.path.join(folder, file)
    unzipname = os.path.splitext(name)[0]
    if overwrite or (not os.path.exists(name) and not os.path.exists(unzipname)):
        get_url_content_timeout(
            url, timeout=timeout, encoding=None, output=name, chunk=2**20
        )
    if unzip and not os.path.exists(unzipname):
        names = ungzip_files(name, unzip=False, where_to=folder)
        os.remove(name)
        if isinstance(names, list):
            if len(names) != 1:
                raise DataException(f"Expecting only one file, not '{names}'")
            return names[0]
        return names
    return name


def download_dump(country, name, folder=".", unzip=True, timeout=-1, overwrite=False):
    """
    Downloads :epkg:`wikipedia dumps`.

    :param country: country
    :param name: name of the stream to download
    :param folder: where to download
    :param unzip: unzip the file
    :param timeout: timeout
    :param overwrite: overwrite
    """
    url = "https://dumps.wikimedia.org/{0}wiki/latest/{0}wiki-{1}".format(country, name)
    file = url.split("/")[-1]
    name = os.path.join(folder, file)
    unzipname = os.path.splitext(name)[0]
    if overwrite or (not os.path.exists(name) and not os.path.exists(unzipname)):
        get_url_content_timeout(
            url, timeout=timeout, encoding=None, output=name, chunk=2**20
        )
    if unzip and not os.path.exists(unzipname):
        names = ungzip_files(name, unzip=False, where_to=folder)
        os.remove(name)
        if isinstance(names, list):
            if len(names) != 1:
                raise DataException(f"Expecting only one file, not '{names}'")
            return names[0]
        return names
    return name[:-3] if name.endswith(".gz") else name


def download_titles(country, folder=".", unzip=True, timeout=-1, overwrite=False):
    """
    Downloads wikipedia titles from
    `dumps.wikimedia.org/frwiki/latest/latest-all-titles-in-ns0.gz
    <https://dumps.wikimedia.org/frwiki/latest/latest-all-titles-in-ns0.gz>`_.

    :param country     country
    :param folder      where to download
    :param unzip       unzip the file
    :param timeout     timeout
    :param overwrite   overwrite
    """
    return download_dump(
        country,
        "latest-all-titles-in-ns0.gz",
        folder,
        unzip=unzip,
        timeout=timeout,
        overwrite=overwrite,
    )


def normalize_wiki_text(text):
    """
    Normalizes a text such as a wikipedia title.

    :param text        text to normalize
    @return                 normalized text
    """
    return text.replace("_", " ").replace("''", '"')


def enumerate_titles(filename, norm=True, encoding="utf8"):
    """
    Enumerates titles from a file.

    :param filename        filename
    :param norm            normalize in the function
    :param encoding        encoding
    """
    if norm:
        with open(filename, "r", encoding=encoding) as f:
            for line in f:
                yield normalize_wiki_text(line.strip(" \r\n\t"))
    else:
        with open(filename, "r", encoding=encoding) as f:
            for line in f:
                yield line.strip(" \r\n\t")
