"""Per-token centering, Hadamard rotation, and symmetric activation quantizer."""

import os

os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from centriq.hadamard import next_power_of_two, sylvester_hadamard

# Fixed normal clip in units of sigma_hat (Sec. 4.2).
CLIP_RATIO = 2.5591
RHO = 0.9995


def _f64(x) -> jax.Array:
    return jnp.asarray(x, dtype=jnp.float64)


def symmetric_qmax(bits: int) -> int:
    """Largest code on the symmetric b-bit grid {-(2^{b-1}-1), ..., 2^{b-1}-1}."""
    bits = int(bits)
    if bits < 1:
        raise ValueError(f"bits must be >= 1, got {bits}")
    return (1 << (bits - 1)) - 1


def center_token(x: jax.Array):
    """Subtract the per-token mean. Returns (centered, mean)."""
    x = _f64(x)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    return x - mean, jnp.squeeze(mean, axis=-1)


def pad_features(x: jax.Array, n_prime: int) -> jax.Array:
    """Zero-pad the last axis up to n_prime. m = 1, so no channel replication."""
    n = x.shape[-1]
    if n_prime < n:
        raise ValueError(f"pad target {n_prime} is smaller than {n}")
    if n_prime == n:
        return x
    pad = [(0, 0)] * (x.ndim - 1) + [(0, n_prime - n)]
    return jnp.pad(x, pad)


def _apply_hadamard(x_pad: jax.Array) -> jax.Array:
    """y = H @ x for each last-axis vector. H is symmetric, so this is x @ H."""
    H = sylvester_hadamard(x_pad.shape[-1])
    return x_pad @ H


def rotate_centered(x: jax.Array) -> jax.Array:
    """Center, pad to the next power of two, and apply H.

    The rotated centered token does not depend on an additive offset c * ones.
    """
    x = _f64(x)
    xc, _ = center_token(x)
    n_prime = next_power_of_two(x.shape[-1])
    return _apply_hadamard(pad_features(xc, n_prime))


def rotate_uncentered(x: jax.Array) -> jax.Array:
    """Pad and apply H without centering. Used to show the mean spike."""
    x = _f64(x)
    n_prime = next_power_of_two(x.shape[-1])
    return _apply_hadamard(pad_features(x, n_prime))


def inverse_rotate_unpad(y: jax.Array, n: int) -> jax.Array:
    """Apply H^{-1} = H and drop padding."""
    y = _f64(y)
    recovered = _apply_hadamard(y)
    return recovered[..., :n]


def gaussian_sigma_hat(y: jax.Array, rho: float = RHO) -> jax.Array:
    """sigma_hat = quantile_rho(|y|) / Phi^{-1}((1+rho)/2)."""
    y = _f64(y)
    q = jnp.quantile(jnp.abs(y), rho, axis=-1)
    phi_inv = jax.scipy.stats.norm.ppf((1.0 + rho) / 2.0)
    return q / phi_inv


def activation_scale(y: jax.Array, bits: int, mode: str = "gaussian", rho: float = RHO) -> jax.Array:
    """Per-token scale for the symmetric grid.

    gaussian: alpha = 2.5591 * sigma_hat / (2^{b-1}-1)
    absmax: alpha = max(|y|) / (2^{b-1}-1), so tests can isolate rounding.
    """
    y = _f64(y)
    qmax = symmetric_qmax(bits)
    if qmax == 0:
        return jnp.zeros(y.shape[:-1], dtype=jnp.float64)
    if mode == "absmax":
        return jnp.max(jnp.abs(y), axis=-1) / jnp.float64(qmax)
    if mode == "gaussian":
        return CLIP_RATIO * gaussian_sigma_hat(y, rho=rho) / jnp.float64(qmax)
    raise ValueError(f"unknown activation scale mode {mode!r}")


def quantize_symmetric(y: jax.Array, scale: jax.Array, bits: int) -> jax.Array:
    """Round onto {-(2^{b-1}-1), ..., 2^{b-1}-1}. Zero scale yields zeros."""
    y = _f64(y)
    scale = _f64(scale)
    qmax = symmetric_qmax(bits)
    qmin = -qmax
    scale_b = scale.reshape(scale.shape + (1,) * (y.ndim - scale.ndim))
    safe = jnp.where(scale_b == 0.0, 1.0, scale_b)
    codes = jnp.clip(jnp.round(y / safe), qmin, qmax)
    return jnp.where(scale_b == 0.0, jnp.zeros_like(codes), codes)


def dequantize_symmetric(codes: jax.Array, scale: jax.Array) -> jax.Array:
    codes = _f64(codes)
    scale = _f64(scale)
    scale_b = scale.reshape(scale.shape + (1,) * (codes.ndim - scale.ndim))
    return codes * scale_b


def activation_codes(x: jax.Array, bits: int, scale_mode: str = "gaussian") -> jax.Array:
    """Codes of the centered, padded, rotated token. Independent of c * ones."""
    y = rotate_centered(x)
    scale = activation_scale(y, bits, mode=scale_mode)
    return quantize_symmetric(y, scale, bits)


def activation_roundtrip(x: jax.Array, bits: int, scale_mode: str = "gaussian") -> jax.Array:
    """Center, rotate, quantize, dequantize, inverse-rotate, unpad, restore the mean."""
    x = _f64(x)
    _, mean = center_token(x)
    y = rotate_centered(x)
    scale = activation_scale(y, bits, mode=scale_mode)
    codes = quantize_symmetric(y, scale, bits)
    y_hat = dequantize_symmetric(codes, scale)
    xc_hat = inverse_rotate_unpad(y_hat, x.shape[-1])
    return xc_hat + mean[..., None]
