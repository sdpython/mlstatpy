# -*- coding: utf-8 -*-
"""
@file
@brief Text normalization
"""
import unicodedata


def remove_diacritics(input_str):
    """
    remove diacritics

    @param      input_str       string to clean
    @return                     cleaned string

    Example::

        enguérand --> enguerand

    .. versionadded:: 1.0
    """
    nkfd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nkfd_form.encode('ASCII', 'ignore')
    return only_ascii.decode("utf8")
