from __future__ import annotations

import importlib

from . import ffi
from . import convertor
from . import utils
from . import quantize
from . import nn
from . import backends


def __getattr__(name: str):
    """Lazily load an optional mobile subsystem.

    Args:
        name: Module attribute requested by the caller.

    Returns:
        The requested optional subsystem module.

    Raises:
        AttributeError: If ``name`` is not a supported optional subsystem.
    """
    if name == "service":
        module = importlib.import_module(f"{__name__}.service")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from .ffi import (
    # Floating point types
    float32,
    float16,
    bfloat16,
    # Signed integer types
    int8,
    int16,
    int32,
    int64,
    # Unsigned integer types
    uint8,
    uint16,
    uint32,
    uint64,
    # Bool type
    boolean,
    # Devices
    cpu,
    cuda,
    qnn,
    # Tensor and utilities
    Tensor,
    empty,
    echo,
    device,
    is_torch_available,
    is_numpy_available,
    from_torch,
    from_numpy,
    zeros,
    ones,
    arange,
    random,
)
from .nn.functional import matmul
