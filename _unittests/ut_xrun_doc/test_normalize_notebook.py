import unittest
import os
import pprint
from nbformat import reader, writes
from nbformat.validator import normalize
from teachpyx import __file__ as teachpyx_file
from teachpyx.ext_test_case import ExtTestCase

VERBOSE = 0
ROOT = os.path.realpath(os.path.abspath(os.path.join(teachpyx_file, "..", "..")))


class TestDocumentationNotebook(ExtTestCase):
    def post_process(self, content):
        lines = []
        for line in content.split("\n"):
            if "get_ipython()" in line:
                line = "# " + line
            lines.append(line)
        return "\n".join(lines)

    def run_test(self, nb_name: str, verbose=0) -> int:
        with open(nb_name, "r", encoding="utf-8") as f:
            content = f.read()
        if "sys.path.append" in content and "module_file_regex.ipynb" not in nb_name:
            raise AssertionError(
                f"'sys.path.append' was found in notebook {nb_name!r}."
            )
        nbdict = reader.reads(content)
        new_dict = normalize(nbdict)
        try:
            new_content = writes(new_dict[1], version=4)
        except AttributeError as e:
            raise AssertionError(
                f"Cannot convert {nb_name!r}\n----\n{pprint.pformat(nbdict)}"
                f"\n-----\n{pprint.pformat(new_dict)}"
            ) from e
        if content != new_content:
            if os.environ.get("NB_NORMALIZE", 0) in (1, "1"):
                if verbose:
                    print(f"[nbformat] normalize {nb_name!r}.")
                with open(nb_name, "w", encoding="utf-8") as f:
                    f.write(new_content)
                    return 1
            raise AssertionError(
                f"Normalization should be run on {nb_name!r}. "
                f"Set NB_NORMALIZE=1 and rerun this file."
            )
        return 1

    @classmethod
    def add_test_methods_path(cls, fold):
        found = os.listdir(fold)
        last = os.path.split(fold)[-1]
        for name in found:
            if name.endswith(".ipynb"):
                fullname = os.path.join(fold, name)

                def _test_(self, fullname=fullname):
                    res = self.run_test(fullname, verbose=VERBOSE)
                    self.assertIn(res, (-1, 1))

                lasts = last.replace("-", "_")
                names = os.path.splitext(name)[0].replace("-", "_")
                setattr(cls, f"test_{lasts}_{names}", _test_)

    @classmethod
    def add_test_methods(cls):
        this = os.path.abspath(os.path.dirname(__file__))
        folds = [
            os.path.join(this, "..", "..", "_doc", "c_data"),
            os.path.join(this, "..", "..", "_doc", "practice", "algo-base"),
            os.path.join(this, "..", "..", "_doc", "practice", "algo-compose"),
            os.path.join(this, "..", "..", "_doc", "practice", "exams"),
            os.path.join(this, "..", "..", "_doc", "practice", "ml"),
            os.path.join(this, "..", "..", "_doc", "practice", "py-base"),
            os.path.join(this, "..", "..", "_doc", "practice", "tds-base"),
            os.path.join(this, "..", "..", "_doc", "practice", "years", "2023"),
        ]
        for fold in folds:
            cls.add_test_methods_path(os.path.normpath(fold))


TestDocumentationNotebook.add_test_methods()

if __name__ == "__main__":
    unittest.main(verbosity=2)
