import os
import stat
import sys
import time
import unittest
import unicodedata
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO, BytesIO
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
        if raise_exception:
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
            except UnicodeDecodeError as e:
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


def remove_folder(top, remove_also_top=True, raise_exception=True):
    """
    Removes everything in folder *top*.

    :param top: path to remove
    :param remove_also_top: remove also root
    :param raise_exception: raise an exception if a file cannot be remove
    :return: list of removed files and folders
        --> list of tuple ( (name, "file" or "dir") )
    """
    if top in {"", "C:", "c:", "C:\\", "c:\\", "d:", "D:", "D:\\", "d:\\"}:
        raise RuntimeError(  # pragma: no cover
            "top is a root (c: for example), this is not safe"
        )

    res = []
    first_root = None
    for root, dirs, files in os.walk(top, topdown=False):
        for name in files:
            t = os.path.join(root, name)
            try:
                os.remove(t)
            except PermissionError as e:  # pragma: no cover
                if raise_exception:
                    raise PermissionError(f"unable to remove file {t}") from e
                remove_also_top = False
                continue
            res.append((t, "file"))
        for name in dirs:
            t = os.path.join(root, name)
            try:
                os.rmdir(t)
            except OSError as e:
                if raise_exception:
                    raise OSError(f"unable to remove folder {t}") from e
                remove_also_top = False  # pragma: no cover
                continue  # pragma: no cover
            res.append((t, "dir"))
        if first_root is None:
            first_root = root

    if top is not None and remove_also_top:
        res.append((top, "dir"))
        os.rmdir(top)

    return res


def get_temp_folder(
    thisfile, name=None, clean=True, create=True, persistent=False, path_name="tpath"
):
    """
    Creates and returns a local temporary folder to store files
    when unit testing.

    :param thisfile: use ``__file__`` or the function which runs the test
    :param name: name of the temporary folder
    :param clean: if True, clean the folder first, it can also a function
        called to determine whether or not the folder should be cleaned
    :param create: if True, creates it (empty if clean is True)
    :param persistent: if True, create a folder at root level to reduce path length,
        the function checks the ``MAX_PATH`` variable and
        shorten the test folder is *max_path* is True on :epkg:`Windows`,
        on :epkg:`Linux`, it creates a folder three level ahead
    :param path_name: test path used when *max_path* is True
    :return: temporary folder

    The function extracts the file which runs this test and will name
    the temporary folder base on the name of the method. *name* must be None.

    Parameter *clean* can be a function.
    Signature is ``def clean(folder)``.
    """
    if name is None:
        name = thisfile.__name__
        if name.startswith("test_"):
            name = "temp_" + name[5:]
        elif not name.startswith("temp_"):
            name = "temp_" + name
        thisfile = os.path.abspath(thisfile.__func__.__code__.co_filename)
    final = os.path.split(name)[-1]

    if not final.startswith("temp_") and not final.startswith("temp2_"):
        raise NameError(f"the folder '{name}' must begin with temp_")

    local = os.path.join(
        os.path.normpath(os.path.abspath(os.path.dirname(thisfile))), name
    )

    if persistent:
        if sys.platform.startswith("win"):
            from ctypes.wintypes import MAX_PATH

            if MAX_PATH <= 300:
                local = os.path.join(os.path.abspath("\\" + path_name), name)
            else:
                local = os.path.join(local, "..", "..", "..", "..", path_name, name)
        else:
            local = os.path.join(local, "..", "..", "..", "..", path_name, name)
        local = os.path.normpath(local)

    if name == local:
        raise NameError(f"The folder '{name}' must be relative, not absolute")

    if not os.path.exists(local):
        if create:
            os.makedirs(local)
            mode = os.stat(local).st_mode
            nmode = mode | stat.S_IWRITE
            if nmode != mode:
                os.chmod(local, nmode)
    else:
        if (callable(clean) and clean(local)) or (not callable(clean) and clean):
            # delayed import to speed up import time of pycode
            remove_folder(local)
            time.sleep(0.1)
        if create and not os.path.exists(local):
            os.makedirs(local)
            mode = os.stat(local).st_mode
            nmode = mode | stat.S_IWRITE
            if nmode != mode:
                os.chmod(local, nmode)

    return local


def noLOG(*args, **kwargs):
    pass


def unzip_files(
    zipf, where_to=None, fLOG=noLOG, fvalid=None, remove_space=True, fail_if_error=True
):
    """
    Unzips files from a zip archive.

    :param zipf: archive (or bytes or BytesIO)
    :param where_to: destination folder (can be None, the result is a list of tuple)
    :param fLOG: logging function
    :param fvalid: function which takes two paths (zip name, local name)
        and return True if the file
        must be unzipped, False otherwise, if None, the default answer is True
    :param remove_space: remove spaces in created local path (+ ``',()``)
    :param fail_if_error: fails if an error is encountered
        (typically a weird character in a filename),
        otherwise a warning is thrown.
    :return: list of unzipped files
    """
    import zipfile

    if isinstance(zipf, bytes):
        zipf = BytesIO(zipf)

    try:
        with zipfile.ZipFile(zipf, "r"):
            pass
    except zipfile.BadZipFile as e:  # pragma: no cover
        if isinstance(zipf, BytesIO):
            raise e
        raise OSError(f"Unable to read file '{zipf}'") from e

    files = []
    with zipfile.ZipFile(zipf, "r") as file:
        for info in file.infolist():
            if fLOG:
                fLOG(f"[unzip_files] unzip '{info.filename}'")
            if where_to is None:
                try:
                    content = file.read(info.filename)
                except zipfile.BadZipFile as e:  # pragma: no cover
                    if fail_if_error:
                        raise zipfile.BadZipFile(
                            f"Unable to extract '{info.filename}' due to {e}"
                        ) from e
                    warnings.warn(
                        f"Unable to extract '{info.filename}' due to {e}",
                        UserWarning,
                        stacklevel=0,
                    )
                    continue
                files.append((info.filename, content))
            else:
                clean = remove_diacritics(info.filename)
                if remove_space:
                    clean = (
                        clean.replace(" ", "")
                        .replace("'", "")
                        .replace(",", "_")
                        .replace("(", "_")
                        .replace(")", "_")
                    )
                tos = os.path.join(where_to, clean)
                if not os.path.exists(tos):
                    if fvalid and not fvalid(info.filename, tos):
                        fLOG("[unzip_files]    skipping", info.filename)
                        continue
                    try:
                        data = file.read(info.filename)
                    except zipfile.BadZipFile as e:  # pragma: no cover
                        if fail_if_error:
                            raise zipfile.BadZipFile(
                                f"Unable to extract '{info.filename}' due to {e}"
                            ) from e
                        warnings.warn(
                            f"Unable to extract '{info.filename}' due to {e}",
                            UserWarning,
                            stacklevel=0,
                        )
                        continue
                    # check encoding to avoid characters not allowed in paths
                    if not os.path.exists(tos):
                        if sys.platform.startswith("win"):
                            tos = tos.replace("/", "\\")
                        finalfolder = os.path.split(tos)[0]
                        if not os.path.exists(finalfolder):
                            fLOG(
                                "[unzip_files]    creating folder (zip)",
                                os.path.abspath(finalfolder),
                            )
                            try:
                                os.makedirs(finalfolder)
                            except FileNotFoundError as e:  # pragma: no cover
                                mes = (
                                    "Unexpected error\ninfo.filename={0}\ntos={1}"
                                    "\nfinalfolder={2}\nlen(nfinalfolder)={3}"
                                ).format(
                                    info.filename, tos, finalfolder, len(finalfolder)
                                )
                                raise FileNotFoundError(mes) from e
                        if not info.filename.endswith("/"):
                            try:
                                with open(tos, "wb") as u:
                                    u.write(data)
                            except FileNotFoundError as e:  # pragma: no cover
                                # probably an issue in the path name
                                # the next lines are just here to distinguish
                                # between the two cases
                                if not os.path.exists(finalfolder):
                                    raise e
                                newname = info.filename.replace(" ", "_").replace(
                                    ",", "_"
                                )
                                if sys.platform.startswith("win"):
                                    newname = newname.replace("/", "\\")
                                tos = os.path.join(where_to, newname)
                                finalfolder = os.path.split(tos)[0]
                                if not os.path.exists(finalfolder):
                                    fLOG(
                                        "[unzip_files]    creating folder (zip)",
                                        os.path.abspath(finalfolder),
                                    )
                                    os.makedirs(finalfolder)
                                with open(tos, "wb") as u:
                                    u.write(data)
                            files.append(tos)
                            fLOG(
                                "[unzip_files]    unzipped ", info.filename, " to ", tos
                            )
                    elif not tos.endswith("/"):  # pragma: no cover
                        files.append(tos)
                elif not info.filename.endswith("/"):  # pragma: no cover
                    files.append(tos)
    return files


def ungzip_files(
    filename,
    where_to=None,
    fLOG=noLOG,
    fvalid=None,
    remove_space=True,
    unzip=True,
    encoding=None,
):
    """
    Uncompresses files from a gzip file.

    :param filename: final gzip file (double compression, extension
        should something like .zip.gz)
    :param where_to: destination folder (can be None, the result is a list of tuple)
    :param fLOG: logging function
    :param fvalid: function which takes two paths (zip name, local name)
        and return True if the file
        must be unzipped, False otherwise, if None, the default answer is True
    :param remove_space: remove spaces in created local path (+ ``',()``)
    :param unzip: unzip file after gzip
    :param encoding: encoding
    :return: list of unzipped files
    """
    import gzip

    if isinstance(filename, bytes):
        is_file = False
        filename = BytesIO(filename)
    else:
        is_file = True

    if encoding is None:
        f = gzip.open(filename, "rb")
        content = f.read()
        f.close()
        if unzip:
            try:
                return unzip_files(content, where_to=where_to, fLOG=fLOG)
            except Exception as e:
                raise OSError(f"Unable to unzip file '{filename}'") from e
        elif where_to is not None:
            filename = os.path.split(filename)[-1].replace(".gz", "")
            filename = os.path.join(where_to, filename)
            with open(filename, "wb") as f:
                f.write(content)
            return filename
        return content
    else:
        f = gzip.open(filename, "rt", encoding="utf-8")
        content = f.read()
        f.close()
        if is_file:
            filename = filename.replace(".gz", "")
            with open(filename, "wb") as f:
                f.write(content)
            return filename
        return content


def remove_diacritics(input_str):
    """
    Removes diacritics.

    :param input_str: string to clean
    :return: cleaned string

    Example::

        enguérand --> enguerand
    """
    nkfd_form = unicodedata.normalize("NFKD", input_str)
    only_ascii = nkfd_form.encode("ASCII", "ignore")
    return only_ascii.decode("utf8")
