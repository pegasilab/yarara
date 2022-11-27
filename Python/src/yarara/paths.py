from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from yarara.sts import spec_time_series

__all__ = ["cwd", "root", "paths"]


@dataclass
class Paths:
    root: str
    generic_mask_ccf: str

    def reinterpolated_mask_ccf(self, sts: spec_time_series) -> Path:
        return Path(sts.dir_root) / "CCF_MASK"

    def __init__(self, root: str) -> None:
        self.root = root
        self.generic_mask_ccf = root + "/Python/MASK_CCF/"

    def sorted_rassine_pickles(self, sts: spec_time_series) -> Sequence[str]:
        files = glob.glob(sts.directory + "RASSI*.p")
        files = np.sort(files)
        return list(files)


cwd = os.getcwd()

paths = Paths(root="/".join(cwd.split("/")[:-1]))

root = paths.root
