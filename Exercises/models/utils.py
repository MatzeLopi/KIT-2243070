from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from scipy import io
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def read_mat(source: str) -> dict:
    """Read a .mat file and return a dictionary with the data

    Args:
        source: path to the .mat file

    Returns:
        data: dictionary with the data
    """
    data = io.loadmat(source)
    for k, v in data.items():
        if k.startswith("__"):
            continue
        try:
            data[k] = jnp.array(v)
        except Exception:
            pass

    return dotdict(data)


@dataclass
class PDEParams:
    d: float
    c: float
    fun: Callable
    t: jax.Array
    u: jax.Array
    z: jax.Array


def build_laplacian_matrix(N: int, dz: float, d: float, dt: float):
    """Build sparse implicit diffusion matrix: (I - dt * d * Dzz)

    Args:
        N (int): Number of grid points.
        dz (float): Grid spacing.
        d (float): Diffusion coefficient.
        dt (float): Time step.

    Returns:
        scipy.sparse.csr_matrix: Implicit diffusion matrix.
    """
    diag = 1 + 2 * d * dt / dz**2
    off = -d * dt / dz**2
    main_diag = diag * jnp.ones(N)
    off_diag = off * jnp.ones(N - 1)
    A = diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format="csr")
    return A


def solve_pde_implicit(p: PDEParams, ic_func=None):
    z = p.z
    dz = z[1] - z[0]
    t = p.t
    dt = t[1] - t[0]
    N = len(z)

    # Initial condition
    if ic_func:
        x = jnp.array([ic_func(zi) for zi in z])
    else:
        x = (
            jnp.sin(jnp.pi * z)
            + jnp.cos(jnp.pi * z)
            + jnp.sin(3 * jnp.pi * z)
            + jnp.cos(3 * jnp.pi * z)
        )

    # Prepare solution storage
    x_hist = [x]

    # Build Laplacian once
    A = build_laplacian_matrix(N, dz, p.d, dt)

    for i in range(len(t) - 1):
        ti = t[i]
        ui = p.u[i]

        # Convection term (explicit)
        dx = (jnp.roll(x, -1) - jnp.roll(x, 1)) / (2 * dz)
        dx = dx.at[0].set(-ui / p.d)  # left BC
        dx = dx.at[-1].set(0.0)  # right BC

        # Reaction term (explicit)
        f_val = jax.vmap(lambda xi, dxi, zi: p.fun(xi, dxi, zi, ti))(x, dx, z)

        rhs = x + dt * (p.c * dx + f_val)

        # Implicit solve: (I - dt*d*Dzz) x_next = rhs
        x = jnp.array(spsolve(A, rhs))  # convert back to JAX array
        x_hist.append(x)

    return jnp.stack(x_hist), z, t


def print_tf(param_vector: jax.Array, na: int, nb: int, var: str = "z"):
    """
    Pretty-print a discrete-time transfer function from a parameter vector.

    Args:
        param_vector (jax.Array): Flattened parameter vector [a₁, ..., a_na, b₁, ..., b_nb]
        na (int): Number of past outputs (denominator order)
        nb (int): Number of past inputs (numerator order)
        var (str): Optional variable symbol. Defaults to "z".
    """
    a = param_vector[:na]
    a = jnp.flip(a)
    b = param_vector[na:]
    b = jnp.flip(b)

    def poly_str(coeffs, offset=1):
        terms = []
        for i, c in enumerate(coeffs):
            power = i + offset
            if abs(c) < 1e-8:
                continue
            c_str = f"{c:.3f}" if abs(c) != 1 else ("-" if c < 0 else "")
            term = f"{c_str}{var}^{power}" if power > 0 else f"{c_str}"
            terms.append(term)
        return " + ".join(terms).replace("+ -", "- ")

    b_str = poly_str(b, offset=1) or "0"
    a_str = "1" + (" + " + poly_str(a, offset=1) if na > 0 else "")

    # Center alignment
    width = max(len(b_str), len(a_str))
    print()
    print(b_str.center(width))
    print("―" * width)
    print(a_str.center(width))
    print()
