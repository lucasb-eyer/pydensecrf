# coding: UTF-8
from distutils.core import setup
from Cython.Build import cythonize

# TODO:
# - Wrap learning.
# - Make LabelCompatibility, UnaryEnergy, PairwisePotential extensible? (Maybe overkill?)

setup(
    name="pydensecrf",
    version="0.1",
    description="A python interface to Philipp Krähenbühl's fully-connected CRF code.",
    author="Lucas Beyer",
    author_email="lucasb.eyer.be@gmail.com",
    url="http://github.com/lucasb-eyer/pydensecrf",
    ext_modules=cythonize(['pydensecrf/eigen.pyx', 'pydensecrf/densecrf.pyx']),
    packages=["pydensecrf"]
)

