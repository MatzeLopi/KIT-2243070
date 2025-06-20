import psutil
import time
import itertools
from functools import partial
from typing import Callable
from dataclasses import dataclass
import jax
import jax.numpy as jnp

from .errors import _mean_absolute_error
from .arx import data_vector as _arx_data_vector

KEY = jax.random.PRNGKey(0)


@dataclass
class HammersteinParams:
    loss: float
    param_vector: jax.Array
    na: int
    nb: int
    order: int
    func: Callable

    def __str__(self):
        return f"Loss: {self.loss}\n Parameters: {self.param_vector},\n na: {self.na},\n nb: {self.nb},\n order: {self.order},\n func: {self.func}"


@partial(jax.jit, static_argnames=["na", "nb", "order", "func"])
def data_vector(
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
    # TODO: Adept this to start apply function to all the u elements
    lin: jax.Array = _arx_data_vector(y, u, na, nb, k)

    sa = lin[:na]
    sb = lin[na:]

    orders = jnp.arange(1, order + 1)

    non_lin = jax.vmap(lambda p: jax.vmap(lambda e: func(e, p))(sb))(orders)

    non_lin = non_lin.reshape(-1)
    return jnp.concatenate([sa, non_lin])


def rsl(
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
        pre_param_vec, prev_uncertanty_mat = rsl(na, nb, k - 1, y, u, order, func)

    sk = jnp.array(data_vector(y, u, na, nb, k, order, func)).reshape(-1, 1)
    k_k = (jnp.dot(prev_uncertanty_mat, sk)) / (
        1 + jnp.dot(sk.T, jnp.dot(prev_uncertanty_mat, sk))
    )
    uncertanty_matrix = prev_uncertanty_mat - jnp.dot(
        k_k, jnp.dot(sk.T, prev_uncertanty_mat)
    )
    param_vector = pre_param_vec + k_k * (y[k] - jnp.dot(sk.T, pre_param_vec))

    return param_vector, uncertanty_matrix


@partial(jax.jit, static_argnames=["na", "nb", "order", "func"])
def fit(
    y: jax.Array,
    u: jax.Array,
    na: int,
    nb: int,
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
        sk = data_vector(y, u, na, nb, k, order, func).reshape(-1, 1)
        k_k = (jnp.dot(prev_param_mat, sk)) / (
            1 + jnp.dot(sk.T, jnp.dot(prev_param_mat, sk))
        )
        uncertanty_matrix = prev_param_mat - jnp.dot(k_k, jnp.dot(sk.T, prev_param_mat))
        param_vector = pre_param_vec + k_k * (y[k] - jnp.dot(sk.T, pre_param_vec))
        return param_vector, uncertanty_matrix

    vec, matrix = jax.lax.fori_loop(0, y.shape[0], body_fun, (init_vector, init_matrix))
    return vec.flatten(), matrix


@partial(jax.jit, static_argnames=["na", "nb", "order", "func"])
def simulate(
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
        s_k = data_vector(y_sim, u, na, nb, k, order, func)
        y_pred = jnp.dot(s_k, param_vector)
        y_sim = y_sim.at[k].set(y_pred)
        return y_sim

    y_sim = jax.lax.fori_loop(init_len, N, body_fun, y_sim)
    return y_sim


@jax.jit
def _polynomial(
    x: jax.Array, p: jax.Array, array: jax.Array = jnp.array([1])
) -> jax.Array:
    """Polynomial function to be used in the Hammerstein model.

    This function evaluates a polynomial of the form:
    x**(p) * coeff[0] + x**(p-1) * coeff[1] + ... + x**(p-n) * coeff[n]

    where p is the order of the polynomial and coeff are the coefficients.
    Args:
        x (jax.Array): Input value
        p (jax.Array): Polynomial order
        array (jax.Array): Coefficients of the polynomial
    Returns:
        jax.Array: Evaluated polynomial
    """
    exponents = p - jnp.arange(array.shape[0])
    powers = x**exponents
    return jnp.sum(powers * array)


def optimize(
    y: jax.Array,
    u: jax.Array,
    na_range: tuple[int, int],
    nb_range: tuple[int, int],
    e_range: tuple[int, int],
    po_range: tuple[int, int],
    pc_range: tuple[float, float],
    *,
    callback: Callable = print,
):
    """Function to optimize the parameters of the Hammerstein model.

    Polynomials will be of the form:
    x**(p) * coeff[0] + x**(p-1) * coeff[1] + ... + x**(p-n) * coeff[n]

    Args:
        y (jax.Array): Output data
        u (jax.Array): Input data
        na_range (tuple[int, int]): Range of na values to test
        nb_range (tuple[int, int]): Range of nb values to test
        e_range (tuple[int, int]): Range of the cascading polynomial exponents to test.
        po_range (tuple[int, int]): Range of polynomial orders to test.
        pc_range (tuple[float, float]): Range of polynomial coefficients to test.
    """

    scalars = jnp.linspace(pc_range[0], pc_range[1], 5)
    arrays = [
        jnp.array(p)
        for i in range(po_range[0], po_range[1] + 1)
        for p in itertools.permutations(scalars, i)
    ]

    funcs = [partial(_polynomial, array=array) for array in arrays]
    orders = range(e_range[0], e_range[1] + 1)

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

        params, *_ = fit(y, u, na, nb, order, funcs[func])
        y_hat = simulate(y[:na], u, na, nb, order, funcs[func], params)
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
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent

            progress = (index + 1) / total_funcs * 100

            callback(
                f"Progress: {progress:.2f} %. Current iteration time: {dt:.2f}s, average: {dt_av:.2f} seconds, CPU: {cpu:.2f}%, RAM: {ram:.2f}%"
            )
            jax.clear_caches()

    obj = HammersteinParams(
        best_loss,
        best_params,
        best_na,
        best_nb,
        best_order,
        best_func,
    )

    return obj


def optimize_grad(
    y: jax.Array,
    u: jax.Array,
    na_range: tuple[int, int],
    nb_range: tuple[int, int],
    e_range: tuple[int, int],
    po_range: tuple[int, int],
    *,
    callback: Callable = print,
):
    """Function to optimize the parameters of the Hammerstein model.

    Polynomials will be of the form:
    x**(p) * coeff[0] + x**(p-1) * coeff[1] + ... + x**(p-n) * coeff[n]

    Args:
        y (jax.Array): Output data
        u (jax.Array): Input data
        na_range (tuple[int, int]): Range of na values to test
        nb_range (tuple[int, int]): Range of nb values to test
        e_range (tuple[int, int]): Range for number of polynomial terms to test.
        po_range (tuple[int, int]): Range of polynomial orders to test.
    """
    orders = range(e_range[0], e_range[1] + 1)

    best_params = None
    best_na = None
    best_nb = None
    best_func = None
    best_order = None
    best_loss = jnp.inf

    na_grid, nb_grid, order_grid, term_grid = jnp.meshgrid(
        jnp.arange(na_range[0], na_range[1]),
        jnp.arange(nb_range[0], nb_range[1]),
        jnp.array(orders),
        jnp.arange(po_range[0], po_range[1]),
        indexing="ij",
    )

    total_params = len(na_grid.flatten())

    params = zip(
        na_grid.flatten(),
        nb_grid.flatten(),
        order_grid.flatten(),
        term_grid.flatten(),
    )

    print(f"Total parameters: {total_params}")
    dt_av = 0
    dt = 0
    for index, (na, nb, order, terms) in enumerate(params):
        start = time.time()

        na = int(na)
        nb = int(nb)
        order = int(order)
        terms = int(terms)

        array = jax.random.uniform(
            KEY, shape=(terms,), minval=0.1, maxval=1.0
        ).flatten()

        for _ in range(25):

            func = partial(
                _polynomial,
                array=array,
            )

            def loss_fn(array):
                func = partial(_polynomial, array=array)
                p_hat, *_ = fit(y, u, na, nb, order, func)
                y_hat = simulate(jnp.zeros((na,)), u, na, nb, order, func, p_hat)
                return _mean_absolute_error(y, y_hat), p_hat

            (loss, p_hat), grad = jax.value_and_grad(loss_fn, has_aux=True)(array)

            if loss < best_loss:
                best_loss = loss
                best_params = p_hat
                best_func = func
                best_order = order
                best_na = na
                best_nb = nb

            if jnp.isnan(grad).any():
                array = jax.random.uniform(
                    KEY, shape=(terms,), minval=0.1, maxval=1.0
                ).flatten()

            else:
                grad_norm = jnp.linalg.norm(grad)
                max_norm = 1.0
                scaled_grad = jnp.where(
                    grad_norm > max_norm, grad * (max_norm / grad_norm), grad
                )

                delta = 10e-6 * scaled_grad
                array = array - delta

                if jnp.all(jnp.abs(delta) < 1e-7):
                    break

            end = time.time()
            dt = end - start
            dt_av = (dt + dt_av) / 2

        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent

        progress = (index + 1) / total_params * 100

        callback(
            f"Progress: {progress:.2f} %. Current iteration time: {dt:.2f}s, average: {dt_av:.2f} seconds, CPU: {cpu:.2f}%, RAM: {ram:.2f}%"
        )
        jax.clear_caches()

    obj = HammersteinParams(
        best_loss,
        best_params,
        best_na,
        best_nb,
        best_order,
        best_func,
    )

    return obj
