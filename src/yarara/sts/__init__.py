import os

from . import (
    activity,
    analysis,
    extract,
    instrument,
    io,
    limbo,
    management,
    outliers,
    processing,
    util,
)

__all__ = [
    "activity",
    "analysis",
    "extract",
    "instrument",
    "io",
    "limbo",
    "management",
    "outliers",
    "processing",
    "util",
]

cwd = os.getcwd()
root = "/".join(cwd.split("/")[:-1])
