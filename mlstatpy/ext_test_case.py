import os
import sys
import unittest
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from timeit import Timer
from typing import Any, Callable, Dict, List, Optional, Union

import numpy
from numpy.testing import assert_allclose


class InternetException(RuntimeError):
    """
    Exception for the function @see fn get_url_content_timeout
    """


def get_url_content_timeout(
    url,
    timeout=10,
    output=None,
    encoding="utf8",
    raise_exception=True,
    chunk=None,
    fLOG=None,
):
    """
    Downloads a file from internet (by default, it assumes
    it is text information, otherwise, encoding should be None).

    :param url: (str) url
    :param timeout: (int) in seconds, after this time,
        the function drops an returns None, -1 for forever
    :param output: (str) if None, the content is stored in that file
    :param encoding: (str) utf8 by default, but if it is None,
        the returned information is binary
    :param raise_exception: (bool) True to raise an exception, False to send a warnings
    :param chunk: (int|None) save data every chunk (only if output is not None)
    :param fLOG: logging function (only applies when chunk is not None)
    :return: content of the url

    If the function automatically detects that the downloaded data is in gzip
    format, it will decompress it.

    The function raises the exception :class:`InternetException`.
    """
    import gzip
    import socket
    import urllib.error as urllib_error
    import urllib.request as urllib_request
    import http.client as http_client

    try:
        from http.client import InvalidURL
    except ImportError:
        InvalidURL = ValueError

    def save_content(content, append=False):
        "local function"
        app = "a" if append else "w"
        if encoding is not None:
            with open(output, app, encoding=encoding) as f:
                f.write(content)
        else:
            with open(output, app + "b") as f:
                f.write(content)

    try:
        if chunk is not None:
            if output is None:
                raise ValueError("output cannot be None if chunk is not None")
            app = [False]
            size = [0]

            def _local_loop(ur):
                while True:
                    res = ur.read(chunk)
                    size[0] += len(res)  # pylint: disable=E1137
                    if fLOG is not None:
                        fLOG("[get_url_content_timeout] downloaded", size, "bytes")
                    if len(res) > 0:
                        if encoding is not None:
                            res = res.decode(encoding=encoding)
                        save_content(res, app)
                    else:
                        break
                    app[0] = True  # pylint: disable=E1137

            if timeout != -1:
                with urllib_request.urlopen(url, timeout=timeout) as ur:
                    _local_loop(ur)
            else:
                with urllib_request.urlopen(url) as ur:
                    _local_loop(ur)
            app = app[0]
            size = size[0]
        else:
            if timeout != -1:
                with urllib_request.urlopen(url, timeout=timeout) as ur:
                    res = ur.read()
            else:
                with urllib_request.urlopen(url) as ur:
                    res = ur.read()
    except (
        urllib_error.HTTPError,
        urllib_error.URLError,
        ConnectionRefusedError,
        socket.timeout,
        ConnectionResetError,
        http_client.BadStatusLine,
        http_client.IncompleteRead,
        ValueError,
        InvalidURL,
    ) as e:
        if raise_exception:
            raise InternetException(f"Unable to retrieve content url='{url}'") from e
        warnings.warn(
            f"Unable to retrieve content from '{url}' because of {e}",
            ResourceWarning,
            stacklevel=0,
        )
        return None
    except Exception as e:
        if raise_exception:  # pragma: no cover
            raise InternetException(
                f"Unable to retrieve content, url='{url}', exc={e}"
            ) from e
        warnings.warn(
            f"Unable to retrieve content from '{url}' "
            f"because of unknown exception: {e}",
            ResourceWarning,
            stacklevel=0,
        )
        raise e

    if chunk is None:
        if len(res) >= 2 and res[:2] == b"\x1f\x8B":
            # gzip format
            res = gzip.decompress(res)

        if encoding is not None:
            try:
                content = res.decode(encoding)
            except UnicodeDecodeError as e:  # pragma: no cover
                # it tries different encoding

                laste = [e]
                othenc = ["iso-8859-1", "latin-1"]

                for encode in othenc:
                    try:
                        content = res.decode(encode)
                        break
                    except UnicodeDecodeError as ee:
                        laste.append(ee)
                        content = None

                if content is None:
                    mes = [f"Unable to parse text from '{url}'."]
                    mes.append("tried:" + str([*encoding, othenc]))
                    mes.append("beginning:\n" + str([res])[:50])
                    for e in laste:
                        mes.append("Exception: " + str(e))
                    raise ValueError("\n".join(mes)) from e
        else:
            content = res
    else:
        content = None

    if output is not None and chunk is None:
        save_content(content)

    return content


def unit_test_going():
    """
    Enables a flag telling the script is running while testing it.
    Avois unit tests to be very long.
    """
    going = int(os.environ.get("UNITTEST_GOING", 0))
    return going == 1


def ignore_warnings(warns: List[Warning]) -> Callable:
    """
    Catches warnings.

    :param warns:   warnings to ignore
    """

    def wrapper(fct):
        if warns is None:
            raise AssertionError(f"warns cannot be None for '{fct}'.")

        def call_f(self):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", warns)
                return fct(self)

        return call_f

    return wrapper


def measure_time(
    stmt: Union[str, Callable],
    context: Optional[Dict[str, Any]] = None,
    repeat: int = 10,
    number: int = 50,
    warmup: int = 1,
    div_by_number: bool = True,
    max_time: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Measures a statement and returns the results as a dictionary.

    :param stmt: string or callable
    :param context: variable to know in a dictionary
    :param repeat: average over *repeat* experiment
    :param number: number of executions in one row
    :param warmup: number of iteration to do before starting the
        real measurement
    :param div_by_number: divide by the number of executions
    :param max_time: execute the statement until the total goes
        beyond this time (approximatively), *repeat* is ignored,
        *div_by_number* must be set to True
    :return: dictionary

    .. runpython::
        :showcode:

        from onnx_extended.ext_test_case import measure_time
        from math import cos

        res = measure_time(lambda: cos(0.5))
        print(res)

    See `Timer.repeat <https://docs.python.org/3/library/
    timeit.html?timeit.Timer.repeat>`_
    for a better understanding of parameter *repeat* and *number*.
    The function returns a duration corresponding to
    *number* times the execution of the main statement.

    .. versionchanged:: 0.4
        Parameter *max_time* was added.
    """
    if not callable(stmt) and not isinstance(stmt, str):
        raise TypeError(
            f"stmt is not callable or a string but is of type {type(stmt)!r}."
        )
    if context is None:
        context = {}

    if isinstance(stmt, str):
        tim = Timer(stmt, globals=context)
    else:
        tim = Timer(stmt)

    if warmup > 0:
        warmup_time = tim.timeit(warmup)
    else:
        warmup_time = 0

    if max_time is not None:
        if not div_by_number:
            raise ValueError(
                "div_by_number must be set to True of max_time is defined."
            )
        i = 1
        total_time = 0
        results = []
        while True:
            for j in (1, 2):
                number = i * j
                time_taken = tim.timeit(number)
                results.append((number, time_taken))
                total_time += time_taken
                if total_time >= max_time:
                    break
            if total_time >= max_time:
                break
            ratio = (max_time - total_time) / total_time
            ratio = max(ratio, 1)
            i = int(i * ratio)

        res = numpy.array(results)
        tw = res[:, 0].sum()
        ttime = res[:, 1].sum()
        mean = ttime / tw
        ave = res[:, 1] / res[:, 0]
        dev = (((ave - mean) ** 2 * res[:, 0]).sum() / tw) ** 0.5
        mes = dict(
            average=mean,
            deviation=dev,
            min_exec=numpy.min(ave),
            max_exec=numpy.max(ave),
            repeat=1,
            number=tw,
            ttime=ttime,
        )
    else:
        res = numpy.array(tim.repeat(repeat=repeat, number=number))
        if div_by_number:
            res /= number

        mean = numpy.mean(res)
        dev = numpy.mean(res**2)
        dev = (dev - mean**2) ** 0.5
        mes = dict(
            average=mean,
            deviation=dev,
            min_exec=numpy.min(res),
            max_exec=numpy.max(res),
            repeat=repeat,
            number=number,
            ttime=res.sum(),
        )

    if "values" in context:
        if hasattr(context["values"], "shape"):
            mes["size"] = context["values"].shape[0]
        else:
            mes["size"] = len(context["values"])
    else:
        mes["context_size"] = sys.getsizeof(context)
    mes["warmup_time"] = warmup_time
    return mes


class ExtTestCase(unittest.TestCase):
    _warns = []

    def assertEndsWith(self, string, suffix):
        if not string.endswith(suffix):
            raise AssertionError(f"{string!r} does not end with {suffix!r}.")

    def assertExists(self, name):
        if not os.path.exists(name):
            raise AssertionError(f"File or folder {name!r} does not exists.")

    def assertEqual(self, *args, **kwargs):
        if isinstance(args[0], numpy.ndarray):
            self.assertEqualArray(*args, **kwargs)
        else:
            super().assertEqual(*args, **kwargs)

    def assertNotEqualArray(
        self,
        expected: numpy.ndarray,
        value: numpy.ndarray,
        atol: float = 0,
        rtol: float = 0,
    ):
        try:
            self.assertEqualArray(expected, value, atol=atol, rtol=rtol)
        except AssertionError:
            return
        raise AssertionError("Both arrays are equal.")

    def assertEqualArray(
        self,
        expected: numpy.ndarray,
        value: numpy.ndarray,
        atol: float = 0,
        rtol: float = 0,
    ):
        self.assertEqual(expected.dtype, value.dtype)
        self.assertEqual(expected.shape, value.shape)
        assert_allclose(expected, value, atol=atol, rtol=rtol)

    def assertAlmostEqual(
        self,
        expected: numpy.ndarray,
        value: numpy.ndarray,
        atol: float = 0,
        rtol: float = 0,
    ):
        if not isinstance(expected, numpy.ndarray):
            expected = numpy.array(expected)
        if not isinstance(value, numpy.ndarray):
            value = numpy.array(value).astype(expected.dtype)
        self.assertEqualArray(expected, value, atol=atol, rtol=rtol)

    def assertRaise(self, fct: Callable, exc_type: Optional[Exception] = None):
        exct = exc_type or Exception
        try:
            fct()
        except exct as e:
            if exc_type is not None and not isinstance(e, exc_type):
                raise AssertionError(f"Unexpected exception {type(e)!r}.")  # noqa: B904
            return
        raise AssertionError("No exception was raised.")

    def assertEmpty(self, value: Any):
        if value is None:
            return
        if len(value) == 0:
            return
        raise AssertionError(f"value is not empty: {value!r}.")

    def assertNotEmpty(self, value: Any):
        if value is None:
            raise AssertionError(f"value is empty: {value!r}.")
        if isinstance(value, (list, dict, tuple, set)):
            if len(value) == 0:
                raise AssertionError(f"value is empty: {value!r}.")

    def assertStartsWith(self, prefix: str, full: str):
        if not full.startswith(prefix):
            raise AssertionError(f"prefix={prefix!r} does not start string  {full!r}.")

    @classmethod
    def tearDownClass(cls):
        for name, line, w in cls._warns:
            warnings.warn(f"\n{name}:{line}: {type(w)}\n  {str(w)}", stacklevel=0)

    def capture(self, fct: Callable):
        """
        Runs a function and capture standard output and error.

        :param fct: function to run
        :return: result of *fct*, output, error
        """
        sout = StringIO()
        serr = StringIO()
        with redirect_stdout(sout), redirect_stderr(serr):
            res = fct()
        return res, sout.getvalue(), serr.getvalue()
