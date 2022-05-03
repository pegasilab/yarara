import warnings

pickle_protocol_version = 3

__version__ = "0.1.0"

# TODO: check what's going on here
import numpy as np

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
