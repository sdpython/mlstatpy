import unittest
import os
import sys
import importlib
import subprocess
import time
from mlstatpy import __file__ as mlstatpy_file
from mlstatpy.ext_test_case import ExtTestCase

VERBOSE = 0
ROOT = os.path.realpath(os.path.abspath(os.path.join(mlstatpy_file, "..", "..")))


def import_source(module_file_path, module_name):
    if not os.path.exists(module_file_path):
        raise FileNotFoundError(module_file_path)
    module_spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    if module_spec is None:
        raise FileNotFoundError(
            "Unable to find '{}' in '{}'.".format(module_name, module_file_path)
        )
    module = importlib.util.module_from_spec(module_spec)
    return module_spec.loader.exec_module(module)


class TestDocumentationExamples(ExtTestCase):
    def run_test(self, fold: str, name: str, verbose=0) -> int:
        ppath = os.environ.get("PYTHONPATH", "")
        if len(ppath) == 0:
            os.environ["PYTHONPATH"] = ROOT
        elif ROOT not in ppath:
            sep = ";" if sys.platform == "win32" else ":"
            os.environ["PYTHONPATH"] = ppath + sep + ROOT
        perf = time.perf_counter()
        try:
            mod = import_source(fold, os.path.splitext(name)[0])
            assert mod is not None
        except FileNotFoundError:
            # try another way
            cmds = [sys.executable, "-u", os.path.join(fold, name)]
            p = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            res = p.communicate()
            _out, err = res
            st = err.decode("ascii", errors="ignore")
            if "No such file or directory" in st:
                raise FileNotFoundError(st)  # noqa: B904
            if len(st) > 0 and "Traceback" in st:
                if '"dot" not found in path.' in st:
                    # dot not installed, this part
                    # is tested in onnx framework
                    if verbose:
                        print(f"failed: {name!r} due to missing dot.")
                    return -1
                raise AssertionError(  # noqa: B904
                    f"Example {name!r} (cmd: {cmds!r} - "
                    f"exec_prefix={sys.exec_prefix!r}) "
                    f"failed due to\n{st}"
                )
        dt = time.perf_counter() - perf
        if verbose:
            print(f"{dt:.3f}: run {name!r}")
        return 1

    @classmethod
    def add_test_methods(cls):
        this = os.path.abspath(os.path.dirname(__file__))
        folds = [
            os.path.normpath(os.path.join(this, "..", "..", "_doc", "examples")),
        ]
        for fold in folds:
            found = os.listdir(fold)
            for name in found:
                if name.startswith("plot_") and name.endswith(".py"):
                    short_name = os.path.split(os.path.splitext(name)[0])[-1]

                    if sys.platform == "win32" and (
                        "protobuf" in name or "td_note_2021" in name
                    ):

                        @unittest.skip("notebook with questions or issues with windows")
                        def _test_(self, name=name, fold=fold):
                            res = self.run_test(fold, name, verbose=VERBOSE)
                            self.assertIn(res, (-1, 1))

                    else:

                        def _test_(self, name=name, fold=fold):
                            res = self.run_test(fold, name, verbose=VERBOSE)
                            self.assertIn(res, (-1, 1))

                    setattr(cls, f"test_{short_name}", _test_)


TestDocumentationExamples.add_test_methods()

if __name__ == "__main__":
    unittest.main(verbosity=2)
