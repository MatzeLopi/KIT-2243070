import jax
import jax.numpy as jnp
from functools import partial

from dataclasses import dataclass
from typing import Callable
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.integrate import solve_ivp


@partial(jax.jit, static_argnames=("r"))
def dmd(
    x: jax.Array, x_prime: jax.Array, r: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Implementation of the Dynamic Mode Decomposition algorithm.

    Args:
        x (jax.Array): Data matrix at n-1.
        x_prime (jax.Array): Data matrix at n.
        r (int): Rank of the reduced order model.

    Returns:
        jax.Array: Reduced order model.
        jax.Array: Dynamics.
        jax.Array: Eigenvalues
    """
    # Step 1
    u, s, vh = jnp.linalg.svd(x, full_matrices=False)
    u_tilde = u[:, 0:r]
    s_tilde = jnp.diag(s[0:r])
    vh_tilde = vh[0:r, :]
    uh_tilde = u_tilde.conj().T

    # Step 2
    vs_inv = jnp.linalg.solve(s_tilde.conj().T, vh_tilde).conj().T
    a_tilde = uh_tilde @ x_prime @ vs_inv

    # Step 3
    lambda_, W = jnp.linalg.eig(a_tilde)
    phi = x_prime @ vs_inv @ W

    return a_tilde, phi, lambda_


@partial(jax.jit, static_argnames=("r"))
def dmdc(x: jax.Array, x_prime: jax.Array, u: jax.Array, r: int):
    """DMDc with model order reduction via truncated SVD.

    Args:
        x (jax.Array): State snapshots at time t (n x m).
        x_prime (jax.Array): State snapshots at time t+1 (n x m).
        u (jax.Array): Control input snapshots (k x m).
        r (int): Truncation rank.

    Returns:
        jax.Array: Reduced-order A matrix (r x r).
        jax.Array: Reduced-order B matrix (r x k).
        jax.Array: Modes (n x r).
    """
    # 1. Compute the SVD of x and extract the leading r modes.
    u_x, _, _ = jnp.linalg.svd(x, full_matrices=False)
    u_tilde = u_x[:, :r]  # n x r
    uh_tilde = u_tilde.conj().T  # r x n

    # 2. Form the combined snapshot matrix omega = [x; u] of size (n+k) x m.
    omega = jnp.vstack((x, u))
    U_omega, s_omega, vh_omega = jnp.linalg.svd(omega, full_matrices=False)

    # 3. Instead of truncating to r, keep r+k singular values to capture control info.
    k = u.shape[0]
    trunc = r + k
    s_omega_tilde = jnp.diag(s_omega[:trunc])
    vh_omega_tilde = vh_omega[:trunc, :]

    # 4. Compute the pseudoinverse (using the truncated SVD factors)
    vs_inv = jnp.linalg.solve(s_omega_tilde.conj().T, vh_omega_tilde).conj().T
    # vs_inv now has shape: m x (r+k)

    # 5. Compute the reduced operator by projecting x_prime into the reduced state space.
    reduced_operator = uh_tilde @ x_prime @ vs_inv  # shape: r x (r+k)

    # 6. Extract the reduced A and B matrices.
    A_tilde = reduced_operator[:, :r]  # shape: r x r
    B_tilde = reduced_operator[:, r : r + k]  # shape: r x k

    # 7. Compute the eigen-decomposition of A_tilde and reconstruct the DMD modes.
    lambda_, W = jnp.linalg.eig(A_tilde)
    phi = x_prime @ vs_inv[:, :r] @ W  # modes, shape: n x r

    return A_tilde, B_tilde, phi, lambda_


@jax.jit
def dmd_reconstruction(
    phi: jax.Array, lambda_: jax.Array, x0: jax.Array, timesteps: jax.Array
):
    """Reconstruct the time dynamics of the DMD model.

    Args:
    phi (jax.Array): DMD modes (n x r).
    lambda_ (jax.Array): DMD eigenvalues (r,).
    x0 (jax.Array): Initial full state (n,).
    timesteps (jax.Array): 1D array of integer timesteps (e.g., [0,1,2,...,T]).

    Returns:
        jax.Array: Reconstructed state trajectory (T+1 x n), with the first row equal to x0.
    """
    # Project initial state onto the reduced coordinates.
    z0 = jnp.linalg.pinv(phi) @ x0  # shape: (r,)

    # Define the reduced dynamics operator as a diagonal matrix from the eigenvalues.
    A_dmd = jnp.diag(lambda_)  # shape: (r x r)

    # Define one time step evolution: z_{k+1} = A_dmd * z_k, then reconstruct x_{k+1} = phi * z_{k+1}.
    def step(carry, _):
        z_prev = carry
        z_next = A_dmd @ z_prev
        x_next = phi @ z_next
        return z_next, x_next

    # Number of steps is one less than the number of timesteps (because x0 is given).
    num_steps = timesteps.shape[0] - 1
    z_final, x_recons = jax.lax.scan(step, z0, jnp.arange(num_steps))

    # Prepend the initial state to the reconstructed trajectory.
    x_hat = jnp.concatenate([x0[jnp.newaxis, :], x_recons], axis=0)
    return x_hat


@jax.jit
def dmdc_reconstruction(x0, u_seq, A_tilde, B_tilde, phi):
    """
    Args:
        x0 (jax.Array): Initial full state (n,).
        u_seq (jax.Array): Control input sequence (T x k) where each row is u_k.
        A_tilde (jax.Array): Reduced A matrix (r x r).
        B_tilde (jax.Array): Reduced B matrix (r x k).
        phi (jax.Array): DMD modes (n x r).

    Returns:
        x_hat (jax.Array): Reconstructed state trajectory (T+1 x n), with x_hat[0] = x0.
    """
    # Compute the reduced initial condition by projecting x0 onto the DMD modes
    z0 = jnp.linalg.pinv(phi) @ x0  # shape: (r,)

    def body_fun(carry, u_k):
        z_prev = carry
        # Propagate the reduced state using the reduced dynamics
        z_next = A_tilde @ z_prev + B_tilde @ u_k  # shape: (r,)
        # Reconstruct full state from reduced state
        x_recon = phi @ z_next  # shape: (n,)
        return z_next, x_recon

    # Iterate over the control sequence using a scan
    # u_seq has shape (T, k)
    z_final, x_recons = jax.lax.scan(body_fun, z0, u_seq)

    # Prepend the initial state x0 to the reconstructed states
    x_hat = jnp.concatenate([x0[jnp.newaxis, :], x_recons], axis=0)
    return x_hat


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
