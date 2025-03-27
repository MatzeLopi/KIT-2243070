from functools import partial

import jax
import jax.numpy as jnp

from .errors import _mean_absolute_error


@partial(jax.jit, static_argnames=("r"))
def fit(x: jax.Array, x_prime: jax.Array, u: jax.Array, r: int):
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
def simulate(x0, u_seq, A_tilde, B_tilde, phi):
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


def optimize(
    x: jax.Array,
    x_prime: jax.Array,
    u: jax.Array,
    t: jax.Array,
    r_range: tuple[float, float] | jax.Array,
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
    best_phi = None
    best_lambda_ = None

    for r in r_range:
        r = int(r)
        a_tilde, b_tilde, phi, lambda_ = fit(x, x_prime, u, r)

        x_recon = simulate(x_0, u.T[:-1], a_tilde, b_tilde, phi).T
        error = _mean_absolute_error(x, x_recon)

        if error < best_error:
            best_error = error
            best_r = r
            best_a_tilde = a_tilde
            best_b_tilde = b_tilde
            best_phi = phi
            best_lambda_ = lambda_

    return best_r, best_a_tilde, best_b_tilde, best_phi, best_lambda_
