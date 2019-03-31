# -*- coding: utf-8 -*-
"""
@brief      test log(time=38s)
"""
import os
import unittest
from pyquickhelper.helpgen import rst2html
from pyquickhelper.pycode import get_temp_folder, skipif_travis, skipif_appveyor, ExtTestCase
from pyquickhelper.filehelper import synchronize_folder


class TestDocPage(ExtTestCase):

    preamble = '''
            \\usepackage{etex}
            \\usepackage{fixltx2e} % LaTeX patches, \\textsubscript
            \\usepackage{cmap} % fix search and cut-and-paste in Acrobat
            \\usepackage[raccourcis]{fast-diagram}
            \\usepackage{titlesec}
            \\usepackage{amsmath}
            \\usepackage{amssymb}
            \\usepackage{amsfonts}
            \\usepackage{graphics}
            \\usepackage{epic}
            \\usepackage{eepic}
            %\\usepackage{pict2e}
            %%% Redefined titleformat
            \\setlength{\\parindent}{0cm}
            \\setlength{\\parskip}{1ex plus 0.5ex minus 0.2ex}
            \\newcommand{\\hsp}{\\hspace{20pt}}
            \\newcommand{\\acc}[1]{\\left\\{#1\\right\\}}
            \\newcommand{\\cro}[1]{\\left[#1\\right]}
            \\newcommand{\\pa}[1]{\\left(#1\\right)}
            \\newcommand{\\R}{\\mathbb{R}}
            \\newcommand{\\HRule}{\\rule{\\linewidth}{0.5mm}}
            %\\titleformat{\\chapter}[hang]{\\Huge\\bfseries\\sffamily}{\\thechapter\\hsp}{0pt}{\\Huge\\bfseries\\sffamily}
            '''.replace("            ", "")

    custom_preamble = """\n
            \\usepackage[all]{xy}
            \\newcommand{\\vecteur}[2]{\\pa{#1,\\dots,#2}}
            \\newcommand{\\N}[0]{\\mathbb{N}}
            \\newcommand{\\indicatrice}[1]{\\mathbf{1\\!\\!1}_{\\acc{#1}}}
            \\newcommand{\\infegal}[0]{\\leqslant}
            \\newcommand{\\supegal}[0]{\\geqslant}
            \\newcommand{\\ensemble}[2]{\\acc{#1,\\dots,#2}}
            \\newcommand{\\fleche}[1]{\\overrightarrow{ #1 }}
            \\newcommand{\\intervalle}[2]{\\left\\{#1,\\cdots,#2\\right\\}}
            \\newcommand{\\independant}[0]
            {\\;\\makebox[3ex]{\\makebox[0ex]{\\rule[-0.2ex]{3ex}{.1ex}}\\!\\!\\!\\!\\makebox[.5ex][l]
            {\\rule[-.2ex]{.1ex}{2ex}}\\makebox[.5ex][l]{\\rule[-.2ex]{.1ex}{2ex}}} \\,\\,}
            \\newcommand{\\esp}{\\mathbb{E}}
            \\newcommand{\\espf}[2]{\\mathbb{E}_{#1}\\pa{#2}}
            \\newcommand{\\var}{\\mathbb{V}}
            \\newcommand{\\pr}[1]{\\mathbb{P}\\pa{#1}}
            \\newcommand{\\loi}[0]{{\\cal L}}
            \\newcommand{\\vecteurno}[2]{#1,\\dots,#2}
            \\newcommand{\\norm}[1]{\\left\\Vert#1\\right\\Vert}
            \\newcommand{\\norme}[1]{\\left\\Vert#1\\right\\Vert}
            \\newcommand{\\dans}[0]{\\rightarrow}
            \\newcommand{\\partialfrac}[2]{\\frac{\\partial #1}{\\partial #2}}
            \\newcommand{\\partialdfrac}[2]{\\dfrac{\\partial #1}{\\partial #2}}
            \\newcommand{\\trace}[1]{tr\\pa{#1}}
            \\newcommand{\\sac}[0]{|}
            \\newcommand{\\abs}[1]{\\left|#1\\right|}
            \\newcommand{\\loinormale}[2]{{\\cal N} \\pa{#1,#2}}
            \\newcommand{\\loibinomialea}[1]{{\\cal B} \\pa{#1}}
            \\newcommand{\\loibinomiale}[2]{{\\cal B} \\pa{#1,#2}}
            \\newcommand{\\loimultinomiale}[1]{{\\cal M} \\pa{#1}}
            \\newcommand{\\variance}[1]{\\mathbb{V}\\pa{#1}}
            \\newcommand{\\scal}[2]{\\left<#1,#2\\right>}
            """.replace("            ", "")

    @skipif_travis("latex is not installed")
    @skipif_appveyor("latex is not installed")
    def test_doc_page(self):
        temp = get_temp_folder(__file__, "temp_doc_page")
        preamble = TestDocPage.preamble + TestDocPage.custom_preamble
        this = os.path.abspath(os.path.dirname(__file__))
        root = os.path.join(this, "..", "..", "_doc",
                            "sphinxdoc", "source", "c_ml")
        image_path = "piecewise"
        rst = os.path.join(root, "piecewise.rst")
        imgs = os.path.join(root, image_path)
        content = self.read_file(rst)
        synchronize_folder(imgs, os.path.join(
            temp, image_path), create_dest=True)

        epkg_dictionary = {
            'XD': 'http://www.xavierdupre.fr',
            'scikit-learn': 'https://scikit-learn.org/stable/',
            'sklearn': ('http://scikit-learn.org/stable/',
                        ('http://scikit-learn.org/stable/modules/generated/{0}.html', 1),
                        ('http://scikit-learn.org/stable/modules/generated/{0}.{1}.html', 2)),
            'ICML 2016': 'link',
        }
        writer = 'html'
        ht = rst2html(content, writer=writer, layout="sphinx", keep_warnings=True,
                      imgmath_latex_preamble=preamble, outdir=temp,
                      epkg_dictionary=epkg_dictionary)
        ht = ht.replace('src="_images/', 'src="')
        ht = ht.replace('/scripts\\bokeh', '../bokeh_plot\\bokeh')
        ht = ht.replace('/scripts/bokeh', '../bokeh_plot/bokeh')
        rst = os.path.join(temp, "out.{0}".format(writer))
        self.write_file(rst, ht)

        ht = ht.split('<div class="section" id="notebooks">')[0]

        # Tests the content.
        self.assertNotIn('runpythonerror', ht)
        lines = ht.split('\n')
        for i, line in enumerate(lines):
            if 'WARNING' in line:
                if "contains reference to nonexisting document" in lines[i + 1]:
                    continue
                else:
                    mes = 'WARNING issue\n  File "{0}", line {1}'.format(
                        rst, i + 1)
                    raise Exception(mes)


if __name__ == "__main__":
    unittest.main()
