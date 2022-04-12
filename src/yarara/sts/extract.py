from __future__ import annotations

import glob as glob
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .. import my_classes as myc
from .. import my_functions as myf

if TYPE_CHECKING:
    from ..my_rassine_tools import spec_time_series


def yarara_get_orders(self: spec_time_series):
    self.import_material()
    mat = self.material
    orders = np.array(mat["orders_rnr"])
    orders = myf.map_rnr(orders)
    orders = np.round(orders, 0)
    self.orders = orders
    return orders


# extract
def yarara_get_pixels(self: spec_time_series):
    self.import_material()
    mat = self.material
    pixels = np.array(mat["pixels_rnr"])
    pixels = myf.map_rnr(pixels)
    pixels = np.round(pixels, 0)
    self.pixels = pixels
    return pixels
