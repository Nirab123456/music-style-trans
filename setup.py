from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("process_sml/audio_adapter.py", compiler_directives={'language_level': "3"}),
)
