import warnings
from typing import Union

pickle_protocol_version = 3

__version__ = "0.1.0"

import numpy as np

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

Float = Union[float, np.float64]
