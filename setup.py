# -*- coding: utf-8 -*-
import sys
import os
from setuptools import find_packages, setup
from pyquicksetup import read_version, read_readme, default_cmdclass
from pyquicksetup.pyquick import _SetupCommand

#########
# settings
#########

project_var_name = "mlstatpy"
versionPython = f"{sys.version_info.major}.{sys.version_info.minor}"
path = "Lib/site-packages/" + project_var_name
readme = "README.rst"
history = "HISTORY.rst"
requirements = None

KEYWORDS = [project_var_name, "Xavier Dupré", "maths", "teachings"]
DESCRIPTION = (
    """Lectures about machine learning, mathematics, statistics, programming."""
)
CLASSIFIERS = [
    "Programming Language :: Python :: 3",
    "Intended Audience :: Developers",
    "Topic :: Scientific/Engineering",
    "Topic :: Education",
    "License :: OSI Approved :: MIT License",
    "Development Status :: 5 - Production/Stable",
]

#######
# data
#######

packages = find_packages()
package_dir = {k: os.path.join(".", k.replace(".", "/")) for k in packages}
package_data = {}


class SetupCommandSphinx(_SetupCommand):
    description = "Builds documentation."

    user_options = [
        ("layout=", None, "format generation, default is html,rst."),
        (
            "nbformats=",
            None,
            "format generation, default is ipynb,slides,html,python,rst,github",
        ),
    ]

    def initialize_options(self):
        self.layout = "html,rst"
        self.nbformats = "ipynb,html,python,rst,github"

    def finalize_options(self):
        pass

    def run(self):
        from pyquickhelper.pycode import process_standard_options_for_setup

        parameters = self.get_parameters()
        parameters["argv"] = ["build_sphinx"]
        parameters["layout"] = self.layout.split(",")
        parameters["nbformats"] = self.nbformats.split(",")
        process_standard_options_for_setup(**parameters)


defcla = default_cmdclass().copy()
defcla["build_sphinx"] = SetupCommandSphinx

setup(
    name=project_var_name,
    version=read_version(__file__, project_var_name),
    author="Xavier Dupré",
    author_email="xavier.dupre@gmail.com",
    license="MIT",
    url=f"http://www.xavierdupre.fr/app/{project_var_name}/helpsphinx/index.html",
    download_url=f"https://github.com/sdpython/{project_var_name}/",
    description=DESCRIPTION,
    long_description=read_readme(__file__),
    cmdclass=defcla,
    keywords=KEYWORDS,
    classifiers=CLASSIFIERS,
    packages=packages,
    package_dir=package_dir,
    package_data=package_data,
    setup_requires=["pyquicksetup>=0.2.3"],
    install_requires=[
        "numpy>=1.16",
        "mlinsights>=0.2",
        "onnxruntime>=1.12",
        "pyquicksetup",
        "scipy>=1.4",
        "skl2onnx",
    ],
)
