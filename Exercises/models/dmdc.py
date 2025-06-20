from functools import partial

import jax
import jax.numpy as jnp

from .errors import _mean_absolute_error


@partial(jax.jit, static_argnames=["r"])
def fit(x, x_prime, u, r):
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
    omega = jnp.vstack((x, u))
    U, S, V = jnp.linalg.svd(omega, full_matrices=False)
    U_r: jax.Array = U[:, :r]
    S_r: jax.Array = jnp.diag(S[:r])
    V_r: jax.Array = V[:r, :]

    energy = jnp.sum(S[:r] ** 2) / jnp.sum(S**2)

    Uhat, _, _ = jnp.linalg.svd(x_prime, full_matrices=False)
    Uhat_r = Uhat[:, :r]

    G_r = x_prime @ V_r.conj().T @ jnp.linalg.inv(S_r) @ U_r.conj().T
    A_r = G_r[:, : x.shape[0]]
    B_r = G_r[:, x.shape[0] :]

    Atilde = Uhat_r.conj().T @ A_r @ Uhat_r
    Btilde = Uhat_r.conj().T @ B_r

    Lambda_tilde, _ = jnp.linalg.eig(Atilde)
    # phi = x_prime @ vs_inv @ W_tilde  # modes, shape: n x r

    return Atilde, Btilde, Lambda_tilde, Uhat_r, energy


@jax.jit
def simulate(x0, u_seq, A_tilde, B_tilde, transform):
    """
    Args:
        x0 (jax.Array): Initial full state (n,).
        u_seq (jax.Array): Control input sequence (T x k) where each row is u_k.
        A_tilde (jax.Array): Reduced A matrix (r x r).
        B_tilde (jax.Array): Reduced B matrix (r x k).
        transform (jax.Array): Transformation matrix (n x r).

    Returns:
        x_hat (jax.Array): Reconstructed state trajectory (T+1 x n), with x_hat[0] = x0.
    """
    # Compute the reduced initial condition by projecting x0 onto the DMD modes
    z0 = jnp.linalg.pinv(transform) @ x0  # shape: (r,)

    def body_fun(carry, u_k):
        z_prev = carry
        # Propagate the reduced state using the reduced dynamics
        z_next = A_tilde @ z_prev + B_tilde @ u_k  # shape: (r,)
        # Reconstruct full state from reduced state
        x_recon = transform @ z_next  # shape: (n,)
        return z_next, x_recon

    # Iterate over the control sequence using a scan
    # u_seq has shape (T, k)
    _, x_recons = jax.lax.scan(body_fun, z0, u_seq)

    # Prepend the initial state x0 to the reconstructed states
    x_hat = jnp.concatenate([x0[jnp.newaxis, :], x_recons], axis=0)
    return x_hat


def optimize(
    x: jax.Array,
    x_prime: jax.Array,
    u: jax.Array,
    r_range: tuple[float, float] | jax.Array,
    *,
    cutoff: None | float = None,
):
    """Optimize the rank of the reduced order model.

    Args:
        x (jax.Array): Data matrix at n-1.
        x_prime (jax.Array): Data matrix at n.
        t (jax.Array): Time vector.
        r_range (tuple[float, float] | jax.Array): Range of ranks to test.

    Returns:
        jax.Array: Optimal rank.
    """
    r_range = jnp.arange(r_range[0], r_range[1])
    x_0 = x[:, 0]
    assert (
        x.shape[0] == x_prime.shape[0]
    ), "x and x_prime must have the same number of rows"
    assert (
        u.shape[1] == x.shape[1]
    ), "x and x_prime must have the same number of columns"

    best_error = jnp.inf
    best_r = None
    best_a_tilde = None
    best_b_tilde = None
    best_lambda_ = None
    best_transform = None
    best_energy = None

    for r in r_range:
        r = int(r)
        a_tilde, b_tilde, lambda_, transform, energy = fit(x, x_prime, u, r)

        x_recon = simulate(x_0, u.T[:-1], a_tilde, b_tilde, transform).T
        error = _mean_absolute_error(x, x_recon)

        if error < best_error:
            best_error = error
            best_r = r
            best_a_tilde = a_tilde
            best_b_tilde = b_tilde
            best_lambda_ = lambda_
            best_transform = transform
            best_energy = energy

        if cutoff is not None and energy < cutoff:
            break

    return (
        best_r,
        best_a_tilde,
        best_b_tilde,
        best_lambda_,
        best_transform,
        best_energy,
    )
