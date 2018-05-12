"""
@brief      test log(time=0s)
"""

import sys
import os
import unittest
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import check_pep8, ExtTestCase

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


class TestCodeStyle(ExtTestCase):
    """Test style."""

    def test_src(self):
        "skip pylint"
        self.assertFalse(src is None)

    def test_style_src(self):
        thi = os.path.abspath(os.path.dirname(__file__))
        src_ = os.path.normpath(os.path.join(thi, "..", "..", "src"))
        check_pep8(src_, fLOG=fLOG,
                   pylint_ignore=('C0103', 'C1801', 'R0201', 'R1705', 'W0108', 'W0613',
                                  'C0111', 'W0201', 'W0212', 'E0203'),
                   skip=["Too many nested blocks",
                         "Module 'numpy.random' has no 'RandomState' member",
                         "Value 'sch' is unsubscriptable",
                         "Instance of 'tuple' has no ",
                         "Instance of '_Stat' has no 'next_nodes' member",
                         "completion.py:125: W0612",
                         "Parameters differ from overridden '",
                         "do not assign a lambda expression, use a def",
                         "Module 'matplotlib.cm' has no 'rainbow' member",
                         "Value 'self.label' is unsubscriptable",
                         "Unused variable 'count_edge_left'",
                         "Unused variable 'k' ",
                         "Redefining built-in 'format'",
                         "poulet.py:146: C0200",
                         ])

    def test_style_test(self):
        thi = os.path.abspath(os.path.dirname(__file__))
        test = os.path.normpath(os.path.join(thi, "..", ))
        check_pep8(test, fLOG=fLOG, neg_pattern="temp_.*",
                   pylint_ignore=('C0103', 'C1801', 'R0201', 'R1705', 'W0108', 'W0613',
                                  'C0111', 'W0212', 'W0212'),
                   skip=["src' imported but unused",
                         "skip_' imported but unused",
                         "skip__' imported but unused",
                         "skip___' imported but unused",
                         "Unused variable 'skip_'",
                         "Unused import src",
                         "Unused variable 'skip_",
                         "imported as skip_",
                         "Imports from package src are not grouped",
                         "Module 'pygame' has no 'init' member",
                         "Module 'pygame' has no 'MOUSEBUTTONUP' member",
                         "test_graph_distance.py:122: W0612",
                         "Instance of 'tuple' has no '",
                         ])


if __name__ == "__main__":
    unittest.main()
