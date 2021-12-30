#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name="tpsim",
    version="1.0.0",
    description="Python code for vectorized test particle simulations",
    url="https://github.com/tien-vo/tpsim.git",
    packages=find_packages(),
    python_requires=">=3.7,<4",
    install_requires=[
        "astropy>=4.0",
        "matplotlib",
        "plasmapy",
        "pathos",
        "tables",
        "numpy",
        "numba",
        "scipy",
    ],
    extras_require={
        "dev": ["pytest", "tox"],
        "test": ["pytest"],
        "mpi": ["mpi4py"],
    },
    entry_points={
        "console_scripts": [],
    },
    scripts=[],
)

