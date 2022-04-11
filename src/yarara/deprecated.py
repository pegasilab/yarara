"""
This modules contains deprecated functions
"""

import warnings
from typing import Any, Mapping


def try_field(dico: Mapping[str, Any], field: str) -> Any:
    warnings.warn("Instead of try_field(dico, field), write dico.get(field)", DeprecationWarning)
    try:
        a = dico[field]
        return a
    except:
        return None
