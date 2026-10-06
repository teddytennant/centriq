"""Seeded float64 checks for the CentriQ reference path."""

import os

os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp
import numpy as np
from numpy.random import default_rng

jax.config.update("jax_enable_x64", True)

from centriq import (
    activation_codes,
    activation_roundtrip,
    activation_scale,
    quantized_linear,
    rods_quantize,
    rotate_centered,
    rotate_uncentered,
    rtn_minmax,
    sylvester_hadamard,
)


def test_hadamard_orthogonal_and_ones():
    for n in (1, 2, 4, 8, 16, 32, 64):
        H = sylvester_hadamard(n)
        assert H.dtype == jnp.float64
        eye = jnp.eye(n, dtype=jnp.float64)
        assert jnp.allclose(H @ H.T, eye, atol=1e-10, rtol=0.0)
        assert jnp.allclose(H.T @ H, eye, atol=1e-10, rtol=0.0)
        image = H @ jnp.ones((n,), dtype=jnp.float64)
        expected = jnp.zeros((n,), dtype=jnp.float64).at[0].set(jnp.sqrt(jnp.float64(n)))
        assert jnp.allclose(image, expected, atol=1e-10, rtol=0.0)
        # Normalized Sylvester entries are +/- 1/sqrt(n).
        assert jnp.allclose(jnp.abs(H), 1.0 / jnp.sqrt(jnp.float64(n)), atol=1e-12, rtol=0.0)


def test_rotated_centered_token_is_offset_invariant():
    rng = default_rng(1)
    x = jnp.asarray(rng.normal(size=40), dtype=jnp.float64)
    y = rotate_centered(x)
    for c in (-4.0, 0.5, 3.25, 12.0):
        y_shift = rotate_centered(x + c)
        assert jnp.allclose(y_shift, y, atol=1e-10, rtol=0.0)


def test_centering_removes_mean_spike():
    rng = default_rng(2)
    n = 64
    # Mean dominates the bulk, so the uncentered Hadamard peak sits on the DC bin.
    x = jnp.asarray(rng.normal(size=n) + 50.0, dtype=jnp.float64)
    peak_raw = jnp.max(jnp.abs(rotate_uncentered(x)))
    peak_centered = jnp.max(jnp.abs(rotate_centered(x)))
    assert peak_raw >= 2.0 * peak_centered


def test_quantized_layer_offset_invariance():
    rng = default_rng(3)
    n, d_out = 24, 6
    x = jnp.asarray(rng.normal(size=n), dtype=jnp.float64)
    W = jnp.asarray(rng.normal(size=(d_out, n)), dtype=jnp.float64)
    b = jnp.asarray(rng.normal(size=d_out), dtype=jnp.float64)
    y = quantized_linear(x, W, b, a_bits=8, scale_mode="gaussian")
    w1 = W @ jnp.ones((n,), dtype=jnp.float64)
    codes = activation_codes(x, bits=8, scale_mode="gaussian")
    for c in (-2.5, 0.75, 4.0, 11.0):
        y_shift = quantized_linear(x + c, W, b, a_bits=8, scale_mode="gaussian")
        # Eq. 6, exact in float64 up to rounding of the shared matmul.
        assert jnp.allclose(y_shift, y + c * w1, atol=1e-9, rtol=0.0)
        codes_shift = activation_codes(x + c, bits=8, scale_mode="gaussian")
        assert jnp.array_equal(codes_shift, codes)


def test_rods_beats_rtn_on_heavy_tail_bulk():
    rng = default_rng(0)
    rows, group = 8, 64
    W_np = rng.normal(0.0, 1.0, size=(rows, group))
    for i in range(rows):
        idx = rng.choice(group, size=3, replace=False)
        signs = rng.choice(np.array([-1.0, 1.0]), size=3)
        W_np[i, idx] = signs * 80.0 * (1.0 + 0.1 * rng.random(3))
    W = jnp.asarray(W_np, dtype=jnp.float64)
    recon_lp = np.asarray(rods_quantize(W, group_size=group))
    recon_rtn = np.asarray(rtn_minmax(W, group_size=group))

    mode = np.median(W_np, axis=1, keepdims=True)
    bulk = np.abs(W_np - mode) <= 2.0
    err_lp = ((W_np - recon_lp) ** 2)[bulk].mean()
    err_rtn = ((W_np - recon_rtn) ** 2)[bulk].mean()
    # ell_p fit must beat min-max RTN on the mode, and it must be a real 4-bit grid.
    assert err_lp < err_rtn
    assert err_lp < 0.5 * err_rtn
    for i in range(rows):
        # s * sigma * (Q - z) has at most 16 distinct values. Identity would not.
        assert np.unique(np.round(recon_lp[i], decimals=8)).size <= 16
        assert np.unique(np.round(recon_lp[i], decimals=8)).size >= 2


def test_eight_bit_roundtrip_beats_two_bit():
    rng = default_rng(4)
    x = jnp.asarray(rng.normal(size=128), dtype=jnp.float64)
    for mode in ("gaussian", "absmax"):
        hat8 = activation_roundtrip(x, bits=8, scale_mode=mode)
        hat2 = activation_roundtrip(x, bits=2, scale_mode=mode)
        err8 = jnp.mean((hat8 - x) ** 2)
        err2 = jnp.mean((hat2 - x) ** 2)
        assert err8 < err2

    y = rotate_centered(x)
    scale = activation_scale(y, bits=8, mode="absmax")
    expected = jnp.max(jnp.abs(y)) / 127.0
    assert jnp.allclose(scale, expected, atol=1e-12, rtol=0.0)
