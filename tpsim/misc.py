"""
    Contains miscellaneous parameters.
"""


__all__ = [
    "basis",
    "filters",
    "logger",
]


import astropy.units as u
import tables as tb
import numpy as np
import logging
import os


####################################
# SI basis
####################################
# Basis for decomposition
basis = [u.kg, u.km, u.s, u.C, u.T, u.rad]


####################################
# pytables variables
####################################
# Use blosc for good performance
filters = tb.Filters(complevel=5, complib="blosc")


####################################
# Logging
####################################
__log_formatter = logging.Formatter(
    fmt="%(asctime)s | %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)


def logger(name, log_file, level=logging.INFO):
    """To set up as manny loggers as needed"""

    fname = os.path.join("logs", f"{log_file}.log")
    if os.path.exists(fname):
        os.remove(fname)

    handler = logging.FileHandler(os.path.join("logs", f"{log_file}.log"))
    handler.setFormatter(__log_formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

