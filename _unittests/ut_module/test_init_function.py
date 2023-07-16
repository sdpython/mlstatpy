"""
@brief      test log(time=2s)
"""
import io
import unittest
from contextlib import redirect_stdout

from pyquickhelper.pycode import ExtTestCase
from mlstatpy import check, _setup_hook


class TestInitFunction(ExtTestCase):
    def test_check(self):
        check()

    def test_hook(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            _setup_hook(True)
        self.assertIn("Success", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
