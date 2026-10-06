"""CentriQ: centering, Hadamard rotation, rank-1 mean restore, grouped ell_p weights."""

import os

os.environ["JAX_ENABLE_X64"] = "1"

import jax

jax.config.update("jax_enable_x64", True)

from centriq.activation import (
    activation_codes,
    activation_roundtrip,
    activation_scale,
    rotate_centered,
    rotate_uncentered,
)
from centriq.hadamard import next_power_of_two, sylvester_hadamard
from centriq.layer import quantized_linear
from centriq.weights import rods_quantize, rtn_minmax

__all__ = [
    "activation_codes",
    "activation_roundtrip",
    "activation_scale",
    "next_power_of_two",
    "quantized_linear",
    "rods_quantize",
    "rotate_centered",
    "rotate_uncentered",
    "rtn_minmax",
    "sylvester_hadamard",
]
