from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("extract_stems.pyx", compiler_directives={'language_level': "3"}),
)
