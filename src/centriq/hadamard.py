"""Normalized Sylvester Hadamard matrices."""

import os

os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def next_power_of_two(n: int) -> int:
    """Smallest power of two at least n. Returns 1 for n <= 1."""
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def sylvester_hadamard(n: int) -> jax.Array:
    """Sylvester Hadamard of order n = 2^k, normalized by 1/sqrt(n).

    H is symmetric and orthogonal. H @ ones puts all energy on the first
    coordinate: sqrt(n) followed by zeros.
    """
    n = int(n)
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"Hadamard order must be a power of two, got {n}")
    H = np.array([[1.0]], dtype=np.float64)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    H = H / np.sqrt(np.float64(n))
    return jnp.asarray(H, dtype=jnp.float64)
