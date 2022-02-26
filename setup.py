from setuptools import setup
import os
from io import open # Py2.7 compatibility

def readme():
    with open(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'README.md'
            ), encoding='utf8') as fp:
        return fp.read()

def get_version(fname):
    with open(fname) as f:
        for line in f:
            if line.startswith("__version__ = '"):
                return line.split("'")[1]
    raise RuntimeError('could not parse version string')

setup(
    name = 'flamp',
    version = get_version('flamp/__init__.py'),
    description = 'Faster linear algebra with multiple precision',
    long_description = readme(),
    long_description_content_type = 'text/markdown',
    author = 'Clemens Hofreither',
    author_email = 'clemens.hofreither@ricam.oeaw.ac.at',
    url = 'https://github.com/c-f-h/flamp',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: BSD License',
    ],
    packages = ['flamp'],
    install_requires = [
        'numpy>=1.11',
        'gmpy2',
    ],
)
