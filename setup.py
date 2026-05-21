#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="morphint",
    version="0.1",
    description="Interpolation of brain morphology between coronal sections using non-linear alignment",
    author="Thomas Funck",
    author_email="thomas.funck@childmind.org",
    url="https://github.com/tfunck/morphint",
    packages=find_packages(),
    python_requires=">=3.8",
)
