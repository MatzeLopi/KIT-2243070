import numpy as np
import jax
import jax.numpy as jnp
from typing import Callable
from functools import partial
import itertools
import time

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
        param_vector = np.zeros(na + nb + 1).reshape(-1, 1)
        uncertanty_mat = np.eye(na + nb + 1) * alpha
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
def arx_fit(
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
        order (int): Order of the static nonlinearity
        func (Callable): Nonlinearity function
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
def arx_sim(
    y0: jax.Array,
    u: jax.Array,
    na: int,
    nb: int,
    param_vector: jax.Array,
) -> jax.Array:
    """Simulate arx model.

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


def arx_optimizer(
    y,
    u,
    na_range,
    nb_range,
):
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
        params, *_ = arx_fit(na, nb, y, u)
        y_hat = arx_sim(y[:na], u, na, nb, params)
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

    return best_params, best_loss, best_na, best_nb


@partial(jax.jit, static_argnames=["na", "nb", "order", "func"])
def hammerstein_data_vector(
    y: jax.Array, u: jax.Array, na: int, nb: int, k: int, order: int, func: Callable
) -> jax.Array:
    """Implementation to create a hammerstein data vector
    Args:
        y (jax.Array):  Vector with output values
        u (jax.Array): Vector with input values
        na (int):
        nb (int):
        k (int): Iteration Index
        order (int): Order of the hammerstein polynomial
        func (Nonlinear function to use for hammerstein)

    Returns:
        jax.Array: Data Vector for step k.
    """
    lin: jax.Array = data_vector(y, u, na, nb, k)
    input_dep = lin[na:]

    orders = jnp.arange(2, order + 1)

    non_lin = jax.vmap(lambda p: jax.vmap(lambda e: func(e, p))(input_dep))(orders)

    non_lin = non_lin.reshape(-1)
    return jnp.concatenate([lin, non_lin])


def rsl_ham(
    na: int,
    nb: int,
    k: int,
    y: jax.Array,
    u: jax.Array,
    order: int,
    func: Callable,
    pre_param_vec: jax.Array | None = None,
    prev_uncertanty_mat: jax.Array | None = None,
    alpha: float = 10e6,
) -> tuple[jax.Array, jax.Array]:
    """Implementation of recursive least squares for a hammerstein model.

    Args:
        na (int): Order of the denominator
        nb (int): Order of the numerator
        k (int): Current index
        y (jax.Array): Output vector
        u (jax.Array): Input vector
        order (int): Order of the static nonlinearity
        func (Callable): Nonlinearity function
        pre_param_vec (jax.Array, Optional): Parameter vector
        prev_param_mat (jax.Array, Optional): Covariance matrix
        alpha (float, Optional): Regularization parameter

    Returns:
        jax.Array: Parameter vector
        jax.Array: Covariance matrix
    """
    if k < 0:
        param_vector = jnp.zeros(na + (nb + 1) * order).reshape(-1, 1)
        uncertanty_matrix = jnp.eye(na + (nb + 1) * order) * alpha
        return param_vector, uncertanty_matrix

    if prev_uncertanty_mat is None or pre_param_vec is None:
        pre_param_vec, prev_uncertanty_mat = rsl_ham(na, nb, k - 1, y, u, order, func)

    sk = jnp.array(hammerstein_data_vector(y, u, na, nb, k, order, func)).reshape(-1, 1)
    k_k = (jnp.dot(prev_uncertanty_mat, sk)) / (
        1 + jnp.dot(sk.T, jnp.dot(prev_uncertanty_mat, sk))
    )
    uncertanty_matrix = prev_uncertanty_mat - jnp.dot(
        k_k, jnp.dot(sk.T, prev_uncertanty_mat)
    )
    param_vector = pre_param_vec + k_k * (y[k] - jnp.dot(sk.T, pre_param_vec))

    return param_vector, uncertanty_matrix


@partial(jax.jit, static_argnames=["na", "nb", "order", "func"])
def hammerstein_jit(
    na: int,
    nb: int,
    y: jax.Array,
    u: jax.Array,
    order: int,
    func: Callable,
    alpha: float = 10e6,
) -> tuple[jax.Array, jax.Array]:
    """Implementation of recursive least squares for a hammerstein model.

    Args:
        na (int): Order of the denominator
        nb (int): Order of the numerator
        y (jax.Array): Output vector
        u (jax.Array): Input vector
        order (int): Order of the static nonlinearity
        func (Callable): Nonlinearity function
        alpha (float, Optional): Regularization parameter

    Returns:
        jax.Array: Parameter vector
        jax.Array: Covariance matrix
    """
    init_vector = jnp.zeros(na + (nb + 1) * order).reshape(-1, 1)
    init_matrix = jnp.eye(na + (nb + 1) * order) * alpha

    def body_fun(k, params):
        pre_param_vec, prev_param_mat = params
        sk = hammerstein_data_vector(y, u, na, nb, k, order, func).reshape(-1, 1)
        k_k = (jnp.dot(prev_param_mat, sk)) / (
            1 + jnp.dot(sk.T, jnp.dot(prev_param_mat, sk))
        )
        uncertanty_matrix = prev_param_mat - jnp.dot(k_k, jnp.dot(sk.T, prev_param_mat))
        param_vector = pre_param_vec + k_k * (y[k] - jnp.dot(sk.T, pre_param_vec))
        return param_vector, uncertanty_matrix

    vec, matrix = jax.lax.fori_loop(0, y.shape[0], body_fun, (init_vector, init_matrix))
    return vec.flatten(), matrix


@partial(jax.jit, static_argnames=["na", "nb", "order", "func"])
def hammerstein_sim(
    y0: jax.Array,
    u: jax.Array,
    na: int,
    nb: int,
    order: int,
    func: Callable,
    param_vector: jax.Array,
) -> jax.Array:
    """Simulate hammerstein model.

    Args:
        y0 (jax.Array): Initial history for the output (length at least na)
        u (jax.Array): Input sequence over simulation horizon.
        na (int): Number of past outputs used in the regressor.
        nb (int): Number of past inputs used.
        order (int): Order of the static nonlinearity (e.g., polynomial order).
        func (Callable): The nonlinearity function to apply (e.g., lambda e, p: e**p).
        param_vector (jax.Array): The identified parameter vector.

    Returns:
        jax.Array: Simulated output sequence.
    """
    N = u.shape[0]
    init_len = y0.shape[0]
    y_sim = jnp.concatenate([y0, jnp.zeros((N - init_len,))])

    def body_fun(k, y_sim):
        s_k = hammerstein_data_vector(y_sim, u, na, nb, k, order, func)
        y_pred = jnp.dot(s_k, param_vector)
        y_sim = y_sim.at[k].set(y_pred)
        return y_sim

    y_sim = jax.lax.fori_loop(init_len, N, body_fun, y_sim)
    return y_sim


@jax.jit
def _polynomial(x, p, array: jax.Array = jnp.array([1])):
    exponents = p - jnp.arange(array.shape[0])
    powers = x**exponents
    return jnp.sum(powers * array)


def hammerstein_optimization(
    y: jax.Array,
    u: jax.Array,
    na_range: tuple[int, int],
    nb_range: tuple[int, int],
    order_range: tuple[int, int],
    polynomial_order_range: tuple[int, int],
    scaler_range: tuple[float, float],
):

    scalars = jnp.linspace(scaler_range[0], scaler_range[1], 5)
    arrays = [
        jnp.array(p)
        for i in range(polynomial_order_range[0], polynomial_order_range[1] + 1)
        for p in itertools.permutations(scalars, i)
    ]

    funcs = [partial(_polynomial, array=array) for array in arrays]
    orders = range(order_range[0], order_range[1] + 1)

    best_params = None
    best_na = None
    best_nb = None
    best_func = None
    best_order = None
    best_loss = jnp.inf

    na_grid, nb_grid, order_grid, func_grid = jnp.meshgrid(
        jnp.arange(na_range[0], na_range[1]),
        jnp.arange(nb_range[0], nb_range[1]),
        jnp.array(orders),
        jnp.arange(len(funcs)),
        indexing="ij",
    )

    na_flat = na_grid.flatten()
    nb_flat = nb_grid.flatten()
    order_flat = order_grid.flatten()
    func_flat = func_grid.flatten()

    print(f"Total parameters: {len(na_flat)}")
    total_funcs = len(na_flat)
    dt_av = 0
    for index, (na, nb, order, func) in enumerate(
        zip(na_flat, nb_flat, order_flat, func_flat)
    ):
        start = time.time()

        na = int(na)
        nb = int(nb)
        order = int(order)
        func = int(func)

        params, *_ = hammerstein_jit(na, nb, y, u, order, funcs[func])
        y_hat = hammerstein_sim(y[:na], u, na, nb, order, funcs[func], params)
        loss = _mean_absolute_error(y, y_hat)

        if loss < best_loss:
            best_loss = loss
            best_params = params
            best_func = funcs[func]
            best_order = order
            best_na = na
            best_nb = nb

        end = time.time()
        dt = end - start
        dt_av = (dt + dt_av) / 2
        if index % 100 == 0:
            progress = (index + 1) / total_funcs * 100

            print(f"Progress: {progress:.2f} %")
            print(f"Current iteration time: {dt:.2f}s, average: {dt_av:.2f} seconds")

    return best_params, best_loss, best_func, best_order, best_na, best_nb
