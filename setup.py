#!/usr/bin/env python

from distutils.core import setup

setup(
    name="morphint",
    version="0.1",
    description="Interpolation of brain morphology between coronal sections using non-linear alignment",
    author="Thomas Funck",
    author_email="thomas.funck@childmind.org",
    url="https://github.com/tfunck/morphint",
    packages=["morphint"],
    python_requires='>3.7.0'
)
