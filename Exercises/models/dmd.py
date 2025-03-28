from functools import partial
import timeit
import jax
import jax.numpy as jnp


from .errors import _mean_absolute_error


@jax.jit
def fit_jit(
    u_tilde: jax.Array, s_tilde: jax.Array, vh_tilde: jax.Array, x_prime: jax.Array
):
    """Jitable part of the DMD algorithm."""

    uh_tilde = u_tilde.conj().T

    # Step 2
    vs_inv = jnp.linalg.solve(s_tilde.conj().T, vh_tilde).conj().T
    a_tilde = uh_tilde @ x_prime @ vs_inv

    # Step 3
    lambda_, W = jnp.linalg.eig(a_tilde)
    phi = x_prime @ vs_inv @ W

    return a_tilde, phi, lambda_, u_tilde


def fit(
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
        jax.Array: Transformation Matrix
    """
    # Step 1
    u, s, vh = jnp.linalg.svd(x, full_matrices=False)
    u_tilde = u[:, 0:r]
    s_tilde = jnp.diag(s[0:r])
    vh_tilde = vh[0:r, :]

    return fit_jit(u_tilde, s_tilde, vh_tilde, x_prime)


@jax.jit
def simulate(
    transformation: jax.Array, lambda_: jax.Array, x0: jax.Array, timesteps: jax.Array
):
    """Reconstruct the time dynamics of the DMD model.

    Args:
    transformation (jax.Array): Transformation matrix.
    lambda_ (jax.Array): DMD eigenvalues.
    x0 (jax.Array): Initial full state.
    timesteps (jax.Array): 1D array of integer timesteps (e.g., [0,1,2,...,T]).

    Returns:
        jax.Array: Reconstructed state trajectory (T+1 x n), with the first row equal to x0.
    """
    z0 = jnp.linalg.pinv(transformation) @ x0

    # Define the reduced dynamics operator as a diagonal matrix from the eigenvalues.
    A_dmd = jnp.diag(lambda_)  # shape: (r x r)

    # Define one time step evolution: z_{k+1} = A_dmd * z_k, then reconstruct x_{k+1} = phi * z_{k+1}.
    def step(carry, _):
        z_prev = carry
        z_next = (A_dmd @ z_prev).real
        x_next = (transformation @ z_next).real
        return z_next, x_next

    # Number of steps is one less than the number of timesteps (because x0 is given).
    num_steps = timesteps.shape[0] - 1
    _, x_recons = jax.lax.scan(step, z0, timesteps[:num_steps])

    # Prepend the initial state to the reconstructed trajectory.
    x_hat = jnp.concatenate([x0[jnp.newaxis, :], x_recons], axis=0)
    return x_hat.real


def optimize(
    x: jax.Array,
    x_prime: jax.Array,
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
        jax.Array: Reduced order model.
        jax.Array: Dynamics.
        jax.Array: Eigenvalues
    """
    r_range = jnp.arange(r_range[0], r_range[1])
    x_0 = x[:, 0]

    best_error = jnp.inf
    best_r = None
    best_a_tilde = None
    best_phi = None
    best_lambda_ = None
    best_transform = None

    for r in r_range:
        r = int(r)
        a_tilde, phi, lambda_, transform = fit(x, x_prime, r)

        x_recon = simulate(transform, lambda_, x_0, t[:-1]).T.real
        error = _mean_absolute_error(x, x_recon)

        if error < best_error:
            best_error = error
            best_r = r
            best_a_tilde = a_tilde
            best_phi = phi
            best_lambda_ = lambda_
            best_transform = transform

    return (
        best_r,
        best_a_tilde,
        best_phi,
        best_lambda_,
        best_transform,
    )
