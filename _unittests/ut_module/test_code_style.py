"""
@brief      test log(time=0s)
"""
import os
import unittest
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import check_pep8, ExtTestCase


class TestCodeStyle(ExtTestCase):
    """Test style."""

    def test_style_src(self):
        thi = os.path.abspath(os.path.dirname(__file__))
        src_ = os.path.normpath(os.path.join(thi, "..", "..", "src"))
        check_pep8(src_, fLOG=fLOG,
                   pylint_ignore=('C0103', 'C1801', 'R0201', 'R1705', 'W0108', 'W0613',
                                  'C0111', 'W0201', 'W0212', 'E0203', 'W0107'),
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
                         "Unable to import 'pygame'",
                         ])

    def test_style_test(self):
        thi = os.path.abspath(os.path.dirname(__file__))
        test = os.path.normpath(os.path.join(thi, "..", ))
        check_pep8(test, fLOG=fLOG, neg_pattern="temp_.*",
                   pylint_ignore=('C0103', 'C1801', 'R0201', 'R1705', 'W0108', 'W0613',
                                  'C0111', 'W0212', 'W0212', 'W0107'),
                   skip=["src' imported but unused",
                         "skip_' imported but unused",
                         "skip__' imported but unused",
                         "skip___' imported but unused",
                         "Unused variable 'skip_'",
                         "imported as skip_",
                         "Module 'pygame' has no 'init' member",
                         "Module 'pygame' has no 'MOUSEBUTTONUP' member",
                         "test_graph_distance.py:122: W0612",
                         "Instance of 'tuple' has no '",
                         "Unable to import 'pygame'",
                         "Unused import src",
                         ])


if __name__ == "__main__":
    unittest.main()
