"""Quantized linear layer: rotated activations, RoDS weights, rank-1 mean restore.

y = W_hat @ Q_a(H P xc) + xbar * w1 + b
w1 = W @ ones, full precision, from the unquantized weight.
W_hat is the RoDS reconstruction of W @ P.T @ H.T.
"""

import os

os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from centriq.activation import (
    activation_scale,
    center_token,
    dequantize_symmetric,
    pad_features,
    quantize_symmetric,
)
from centriq.hadamard import next_power_of_two, sylvester_hadamard
from centriq.weights import rods_quantize


def _f64(x) -> jax.Array:
    return jnp.asarray(x, dtype=jnp.float64)


def quantized_linear(
    x: jax.Array,
    W: jax.Array,
    b: jax.Array | None = None,
    a_bits: int = 8,
    group_size: int = 64,
    scale_mode: str = "gaussian",
) -> jax.Array:
    """Eq. 5. Offset invariant: y(x + c * ones) = y(x) + c * w1."""
    x = _f64(x)
    W = _f64(W)
    if W.ndim != 2:
        raise ValueError(f"W must have shape (d_out, n_in), got {W.shape}")
    n = W.shape[1]
    if x.shape[-1] != n:
        raise ValueError(f"x has n={x.shape[-1]} but W has n={n}")

    n_prime = next_power_of_two(n)
    H = sylvester_hadamard(n_prime)
    # P zero-pads, so W @ P.T is W with zero columns, then rotate by H.T.
    # H is symmetric, so @ H.T is @ H.
    W_rot = pad_features(W, n_prime) @ H
    W_hat = rods_quantize(W_rot, group_size=group_size)

    xc, xbar = center_token(x)
    y_rot = pad_features(xc, n_prime) @ H
    scale = activation_scale(y_rot, a_bits, mode=scale_mode)
    codes = quantize_symmetric(y_rot, scale, a_bits)
    q = dequantize_symmetric(codes, scale)

    y = q @ W_hat.T
    w1 = W @ jnp.ones((n,), dtype=jnp.float64)
    y = y + xbar[..., None] * w1
    if b is not None:
        y = y + _f64(b)
    return y
