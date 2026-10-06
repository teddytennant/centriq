"""RoDS grouped 4-bit weight quantizer (Eq. 7, Eq. 8).

Reconstruction of row i, group g is s_i * sigma_ig * (Q_ig - z_ig), with
codes in {0, ..., 15}. sigma_ig is the min-max step (range / 15) and is not
updated. s is a per-row scale shared across groups, initialized at 1 so the
first inner rounding matches min-max RTN. s is fit by IRLS under ell_p.
The inner loop refits codes and zero-points on W_g / s by half-quadratic
splitting with an auxiliary residual E.
"""

import os

os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

QMAX = 15
P_NORM = 0.7
BETA = 10.0
GROUP_SIZE = 64
IRLS_EPS = 1e-8
MAX_OUTER = 3
MAX_IRLS = 5
MAX_PROX = 20
REL_TOL = 1e-3


def _f64(x) -> jax.Array:
    return jnp.asarray(x, dtype=jnp.float64)


def shrink_lp(residual: jax.Array, beta: float = BETA, p: float = P_NORM) -> jax.Array:
    """One-step HQQ generalized shrinkage, a proximal step for ||.||_p^p.

    E = sign(r) * relu(|r| - (1/beta) * |r|^{p-1}).
    For p = 1 this is the soft-threshold of the half-quadratic penalty beta.
    """
    residual = _f64(residual)
    magnitude = jnp.maximum(jnp.abs(residual), 1e-12)
    threshold = (1.0 / beta) * jnp.power(magnitude, p - 1.0)
    shrunk = jnp.maximum(jnp.abs(residual) - threshold, 0.0)
    return jnp.sign(residual) * shrunk


def _group_bounds(n_in: int, group_size: int):
    start = 0
    while start < n_in:
        end = min(start + group_size, n_in)
        yield start, end
        start = end


def _rel_change(new: jax.Array, old: jax.Array) -> jax.Array:
    return jnp.max(jnp.abs(new - old) / (jnp.abs(old) + 1e-12))


def _minmax_params(group: jax.Array):
    """Fixed min-max step and the RTN zero point. sigma stays at this value."""
    row_max = jnp.max(group, axis=-1)
    row_min = jnp.min(group, axis=-1)
    sigma = (row_max - row_min) / jnp.float64(QMAX)
    sigma_safe = jnp.where(sigma == 0.0, 1.0, sigma)
    z = jnp.where(sigma == 0.0, 0.0, -row_min / sigma_safe)
    return sigma, sigma_safe, z


def rtn_minmax(W: jax.Array, group_size: int = GROUP_SIZE) -> jax.Array:
    """Per-group per-row min-max RTN. recon = sigma * (Q - z), Q in {0..15}."""
    W = _f64(W)
    if W.ndim != 2:
        raise ValueError(f"W must have shape (d_out, n_in), got {W.shape}")
    parts = []
    for start, end in _group_bounds(W.shape[1], group_size):
        group = W[:, start:end]
        sigma, sigma_safe, z = _minmax_params(group)
        codes = jnp.clip(jnp.round(group / sigma_safe[:, None] + z[:, None]), 0.0, QMAX)
        recon = sigma[:, None] * (codes - z[:, None])
        recon = jnp.where(sigma[:, None] == 0.0, group, recon)
        parts.append(recon)
    return jnp.concatenate(parts, axis=1)


def rods_quantize(
    W: jax.Array,
    group_size: int = GROUP_SIZE,
    p: float = P_NORM,
    beta: float = BETA,
    eps: float = IRLS_EPS,
    max_outer: int = MAX_OUTER,
    max_irls: int = MAX_IRLS,
    max_prox: int = MAX_PROX,
    tol: float = REL_TOL,
) -> jax.Array:
    """Quantize a rotated weight with RoDS and return the float reconstruction.

    Caps: outer <= 3, IRLS <= 5, proximal <= 20, relative tolerance 1e-3 on
    the scale updates. The proximal budget is the full 20 HQQ steps: each
    zero-point step moves by less than 1e-3, and the cumulative shift is what
    aligns the grid with the mode. Group scale is not refit.
    """
    W = _f64(W)
    if W.ndim != 2:
        raise ValueError(f"W must have shape (d_out, n_in), got {W.shape}")
    d_out, n_in = W.shape
    bounds = list(_group_bounds(n_in, group_size))
    groups = [W[:, start:end] for start, end in bounds]
    sigmas = []
    sigma_safe = []
    zeros = []
    for group in groups:
        sigma, safe, z = _minmax_params(group)
        sigmas.append(sigma)
        sigma_safe.append(safe)
        zeros.append(z)

    s = jnp.ones((d_out,), dtype=jnp.float64)
    codes = [jnp.zeros_like(group) for group in groups]

    for _outer in range(max_outer):
        s_enter = s
        for g, group in enumerate(groups):
            sigma = sigmas[g]
            safe = sigma_safe[g]
            z = zeros[g]
            active = sigma > 0.0
            # Preconditioned tile. s starts at 1, so the first rounding is RTN.
            U = group / s[:, None]
            for _it in range(max_prox):
                codes_g = jnp.clip(jnp.round(U / safe[:, None] + z[:, None]), 0.0, QMAX)
                recon_u = safe[:, None] * (codes_g - z[:, None])
                residual = U - recon_u
                aux = shrink_lp(residual, beta=beta, p=p)
                # Per-row mean offset between codes and the residual-corrected target.
                corrected = (U - aux) / safe[:, None]
                z_new = jnp.mean(codes_g - corrected, axis=-1)
                z = jnp.where(active, z_new, z)
            codes[g] = codes_g
            zeros[g] = z

        for _irls in range(max_irls):
            numer = jnp.zeros((d_out,), dtype=jnp.float64)
            denom = jnp.zeros((d_out,), dtype=jnp.float64)
            for g, group in enumerate(groups):
                sigma = sigmas[g]
                v = sigma[:, None] * (codes[g] - zeros[g][:, None])
                residual = group - s[:, None] * v
                # IRLS weights (eps + r^2)^((p-2)/2). Inactive (zero-range) rows drop out.
                omega = jnp.power(eps + residual ** 2, (p - 2.0) / 2.0)
                omega = jnp.where(sigma[:, None] > 0.0, omega, 0.0)
                numer = numer + jnp.sum(omega * group * v, axis=-1)
                denom = denom + jnp.sum(omega * v ** 2, axis=-1)
            s_new = jnp.where(denom > 1e-18, numer / denom, s)
            bad = (~jnp.isfinite(s_new)) | (s_new <= 0.0)
            s_new = jnp.where(bad, s, s_new)
            rel = _rel_change(s_new, s)
            s = s_new
            if float(rel) < tol:
                break
        if float(_rel_change(s, s_enter)) < tol:
            break

    parts = []
    for g, group in enumerate(groups):
        sigma = sigmas[g]
        recon = s[:, None] * sigma[:, None] * (codes[g] - zeros[g][:, None])
        recon = jnp.where(sigma[:, None] == 0.0, group, recon)
        parts.append(recon)
    return jnp.concatenate(parts, axis=1)
