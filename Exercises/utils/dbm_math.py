import numpy as np
import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnames=["na", "nb"])
def data_vector(y: jax.Array, u: jax.Array, na: int, nb: int, k: int) -> jax.Array:
    """Function to construct the data vector for an Autoregressive Model.

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


def hammerstein_data_vector():
    # TODO: Implement a Hammerstein data vector
    pass


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


def least_squares(
    na: int,
    nb: int,
    k: int,
    y: jax.Array,
    u: jax.Array,
    pre_param_vec: jax.Array | None = None,
    prev_param_mat: jax.Array | None = None,
    alpha: float = 10e6,
) -> tuple[jax.Array, jax.Array]:
    """Implementation of recursive least squares."""
    # TODO: Make it jit
    if k < 0:
        param_vector = np.zeros(na + nb + 1).reshape(-1, 1)
        param_matrix = np.eye(na + nb + 1) * alpha
        return param_vector, param_matrix

    if prev_param_mat is None or pre_param_vec is None:
        pre_param_vec, prev_param_mat = rec_least_squares(na, nb, k - 1, y, u)

    sk = jnp.array(data_vector(y, u, na, nb, k)).reshape(-1, 1)
    k_k = (jnp.dot(prev_param_mat, sk)) / (
        1 + jnp.dot(sk.T, jnp.dot(prev_param_mat, sk))
    )
    param_matrix = prev_param_mat - jnp.dot(k_k, jnp.dot(sk.T, prev_param_mat))
    param_vector = pre_param_vec + k_k * (y[k] - jnp.dot(sk.T, pre_param_vec))

    return param_vector, param_matrix


def rec_least_squares(
    n_a: int,
    n_b: int,
    k: int,
    y: np.array,
    u: np.array,
    p_hat_k_1: np.array = None,
    P_k_1: np.array = None,
    alpha: float = 10e6,
) -> tuple[np.array, np.array]:
    """Recursive least squares algorithm.
    p_hat_k_1 and P_k_1 are optional parameters.
    If they are not provided, they are recursivly calculated.

    Args:
        n_a (int): Order of the denominator
        n_b (int): Order of the numerator
        k (int): Current index
        y (np.array): Output vector
        u (np.array): Input vector
        p_hat_k_1 (np.array, Optional): Parameter vector
        P_k_1 (np.array, Optional): Covariance matrix


    Returns:
        p_k (np.array): Parameter vector
        P_k (np.array): Covariance matrix
    """
    if k < 0:
        param_vector = np.zeros(n_a + n_b + 1).reshape(-1, 1)
        param_matrix = np.eye(n_a + n_b + 1) * alpha
        return param_vector, param_matrix

    if P_k_1 is None or p_hat_k_1 is None:
        p_hat_k_1, P_k_1 = rec_least_squares(n_a, n_b, k - 1, y, u)

    s_k = np.array(data_vector(y, u, n_a, n_b, k)).reshape(-1, 1)
    k_k = (np.dot(P_k_1, s_k)) / (1 + np.dot(s_k.T, np.dot(P_k_1, s_k)))
    P_K = P_k_1 - np.dot(k_k, np.dot(s_k.T, P_k_1))
    p_hat_k = p_hat_k_1 + k_k * (y[k] - np.dot(s_k.T, p_hat_k_1))

    return p_hat_k, P_K
