#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension

setup(name="ckm",
    ext_modules=[
        Extension("ckm", ["constrained_k_means.cpp", "utils.cpp", "point.cpp"],
        extra_compile_args=['-std=c++17'],
        libraries = ["boost_python3"])
    ])

