import time
from functools import partial

import jax
import jax.numpy as jnp

if __name__ != "__main__":
    from .errors import _mean_absolute_error


@partial(jax.jit, static_argnames=["na", "nb"])
def data_vector(y: jax.Array, u: jax.Array, na: int, nb: int, k: int) -> jax.Array:
    """Function to construct the data vector for an ARX model.

    Args:
        y (jax.Array):  Vector with output values
        u (jax.Array): Vector with input values
        na (int):
        nb (int):
        k (int): Iteration Index

    Returns:
        jax.Array: Data Vector for step k.
    """
    sa_start = jnp.maximum(0, k - na)
    sb_start = jnp.maximum(0, k - nb)

    sa_len = k - sa_start
    sb_len = k - sb_start

    sa_mask = jnp.where(jnp.arange(1, na + 1) < (sa_len + 1), 1, 0)
    sb_mask = jnp.where(jnp.arange(1, nb + 2) < (sb_len + 2), 1, 0).at[0].set(1)
    sa_slice = jax.lax.dynamic_slice(y, (sa_start,), na) * sa_mask
    sb_slice = jax.lax.dynamic_slice(u, (sb_start,), nb + 1) * sb_mask

    sa = jnp.flip(sa_slice)
    sb = jnp.flip(sb_slice)
    sa = jnp.roll(sa, sa_len - na)
    sb = jnp.roll(sb, sb_len - nb)

    s = jnp.concatenate([sa, sb])
    return s


@partial(jax.jit, static_argnames=["na", "nb"])
def data_matrix(y: jax.Array, u: jax.Array, na: int, nb: int) -> jax.Array:
    """Function to create data matrix from all inputs and outputs.
    Y and U need to have same number of rows.

    Args:
        y (jax.Array): Array containing the output data
        u (jax.Array): Array containing the input data
        na (int): Size of na vector
        nb (int): Size of nb vector

    Returns:
        jax.Array: Data Matrix with y.shape[0] rows. Can be used to estimate parameters for ARX model.

    """
    ks = jnp.arange(y.shape[0])
    vectorized = jax.vmap(lambda k: data_vector(y, u, na, nb, k))
    return vectorized(ks)


def rls(
    na: int,
    nb: int,
    k: int,
    y: jax.Array,
    u: jax.Array,
    pre_param_vec: jax.Array | None = None,
    prev_uncertanty_mat: jax.Array | None = None,
    alpha: float = 10e6,
) -> tuple[jax.Array, jax.Array]:
    """Implementation of recursive least squares.

    Args:
        na (int): Order of the denominator
        nb (int): Order of the numerator
        k (int): Current index
        y (jax.Array): Output vector
        u (jax.Array): Input vector
        pre_param_vec (jax.Array, Optional): Parameter vector
        prev_param_mat (jax.Array, Optional): Covariance matrix
        alpha (float, Optional): Regularization parameter

    Returns:
        jax.Array: Parameter vector
        jax.Array: Covariance matrix
    """
    if k < 0:
        param_vector = jnp.zeros(na + nb + 1).reshape(-1, 1)
        uncertanty_mat = jnp.eye(na + nb + 1) * alpha
        return param_vector, uncertanty_mat

    if prev_uncertanty_mat is None or pre_param_vec is None:
        pre_param_vec, prev_uncertanty_mat = rls(na, nb, k - 1, y, u)

    sk = data_vector(y, u, na, nb, k).reshape(-1, 1)
    k_k = (jnp.dot(prev_uncertanty_mat, sk)) / (
        1 + jnp.dot(sk.T, jnp.dot(prev_uncertanty_mat, sk))
    )
    uncertanty_mat = prev_uncertanty_mat - jnp.dot(
        k_k, jnp.dot(sk.T, prev_uncertanty_mat)
    )
    param_vector = pre_param_vec + k_k * (y[k] - jnp.dot(sk.T, pre_param_vec))

    return param_vector, uncertanty_mat


@partial(jax.jit, static_argnames=["na", "nb"])
def fit(
    na: int,
    nb: int,
    y: jax.Array,
    u: jax.Array,
    alpha: float = 10e6,
) -> tuple[jax.Array, jax.Array]:
    """Implementation of recursive least squares for a hammerstein model.

    Args:
        na (int): Order of the denominator
        nb (int): Order of the numerator
        y (jax.Array): Output vector
        u (jax.Array): Input vector
        alpha (float, Optional): Regularization parameter

    Returns:
        jax.Array: Parameter vector
        jax.Array: Covariance matrix
    """
    init_vector = jnp.zeros(na + nb + 1).reshape(-1, 1)
    init_matrix = jnp.eye(na + nb + 1) * alpha

    def body_fun(k, carry):
        pre_param_vec, prev_uncertanty_mat = carry
        sk: jax.Array = data_vector(y, u, na, nb, k).reshape(-1, 1)
        k_k: jax.Array = (jnp.dot(prev_uncertanty_mat, sk)) / (
            1 + jnp.dot(sk.T, jnp.dot(prev_uncertanty_mat, sk))
        )
        uncertanty_mat = prev_uncertanty_mat - jnp.dot(
            k_k, jnp.dot(sk.T, prev_uncertanty_mat)
        )
        param_vector = pre_param_vec + k_k * (y[k] - jnp.dot(sk.T, pre_param_vec))

        return (param_vector, uncertanty_mat)

    vec, matrix = jax.lax.fori_loop(0, y.shape[0], body_fun, (init_vector, init_matrix))
    return vec.flatten(), matrix


@partial(jax.jit, static_argnames=["na", "nb"])
def simulate(
    y0: jax.Array,
    u: jax.Array,
    na: int,
    nb: int,
    param_vector: jax.Array,
) -> jax.Array:
    """Simulate arx model.

    This function simulates the output of a discrete-time ARX (AutoRegressive with eXogenous input) model,
    given an input sequence and a parameter vector identified, for example, using Recursive Least Squares (RLS).

    The regressor is constructed from past outputs and past inputs, and multiplied with the parameter vector
    to compute the current output.

    This is mathematically equivalent to applying a discrete-time transfer function of the form:

        G(q⁻¹) = B(q⁻¹) / A(q⁻¹)

    where:
        A(q⁻¹) = 1 + a₁ q⁻¹ + a₂ q⁻² + ... + a_na q⁻ⁿᵃ
        B(q⁻¹) = b₁ q⁻¹ + b₂ q⁻² + ... + b_nb q⁻ⁿᵇ

    After rearranging the difference equation, the system can be expressed in predictive form:

        yₖ = -a₁ yₖ₋₁ - ... - a_na yₖ₋ₙₐ + b₁ uₖ₋₁ + ... + b_nb uₖ₋ₙᵦ

    This structure is directly reflected in the construction of the `data_vector(...)`, making this simulation
    numerically identical to evaluating the transfer function in the time domain.

    Args:
        y0 (jax.Array): Initial history for the output (length at least na)
        u (jax.Array): Input sequence over simulation horizon.
        na (int): Number of past outputs used in the regressor.
        nb (int): Number of past inputs used.
        param_vector (jax.Array): The identified parameter vector.

    Returns:
        jax.Array: Simulated output sequence.
    """
    assert param_vector.ndim == u.ndim
    N = u.shape[0]
    init_len = y0.shape[0]
    y_sim = jnp.concatenate([y0, jnp.zeros((N - init_len,))])

    def body_fun(k, y_sim):
        s_k = data_vector(y_sim, u, na, nb, k)
        y_pred = jnp.dot(s_k, param_vector)
        y_sim = y_sim.at[k].set(y_pred)
        return y_sim

    y_sim = jax.lax.fori_loop(init_len, N, body_fun, y_sim)
    return y_sim


def optimize(
    y: jax.Array,
    u: jax.Array,
    na_range: jax.Array | tuple[int, int],
    nb_range: jax.Array | tuple[int, int],
) -> tuple[jax.Array, jax.Array, int, int]:
    """Function to optimize the parameters of the ARX model.

    Args:
        y (jax.Array): Output data
        u (jax.Array): Input data
        na_range (tuple[int, int]): Range of na values to test
        nb_range (tuple[int, int]): Range of nb values to test

    Returns:
        jax.Array: Best parameter vector
        jax.Array: Best loss
        int: Best na value
        int: Best nb value
    """

    nas = jnp.arange(na_range[0], na_range[1])
    nbs = jnp.arange(nb_range[0], nb_range[1])

    na_grid, nb_grid = jnp.meshgrid(nas, nbs, indexing="ij")
    na_flat = na_grid.flatten()
    nb_flat = nb_grid.flatten()

    best_params = None
    best_na = None
    best_nb = None
    best_loss = jnp.inf

    total_params = len(na_flat)
    dt_av = 0

    print(f"Testing {total_params} combinations.")
    for index, (na, nb) in enumerate(zip(na_flat, nb_flat)):
        start = time.time()
        na = int(na)
        nb = int(nb)
        params, *_ = fit(na, nb, y, u)
        y_hat = simulate(y[:na], u, na, nb, params)
        loss = _mean_absolute_error(y, y_hat)

        if loss < best_loss:
            best_loss = loss
            best_params = params
            best_na = na
            best_nb = nb

        end = time.time()

        dt = end - start
        dt_av = (dt + dt_av) / 2

        if index % 100 == 0:
            progress = (index + 1) / total_params * 100
            print(f"Progress: {progress:.2f} %")
            print(f"Current iteration: {index + 1} / {total_params}")
            print(
                f"Iteration took {end - start:.2f} seconds, average {dt_av:.2f} seconds"
            )
            jax.clear_caches()

    return best_params, best_loss, best_na, best_nb
