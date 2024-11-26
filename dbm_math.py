import numpy as np


def calc_s_k(y: np.array, u: np.array, n_a, n_b, k) -> list:
    """Construct the s_k vector

    Args:
        y (np.array): Output vector
        u (np.array): Input vector
        n_a (int): Order of the denominator
        n_b (int): Order of the numerator
        k (int): Current index

    Returns:
        list: s_k vector
    """

    s_a = list(np.zeros(n_a))
    s_b = list(np.zeros(n_b + 1))

    for i in range(k + 1):
        index_a: int = k - i - 1

        index_b: int = k - i

        if index_a >= 0:
            try:
                s_a[index_a] = float(y[i])
            except Exception:
                pass
        if index_b >= 0:
            try:
                s_b[index_b] = float(u[i])
            except Exception as e:
                pass

    s = s_a + s_b
    return s


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
        p_init = np.zeros(n_a + n_b + 1).reshape(-1, 1)
        P_init = np.eye(n_a + n_b + 1) * alpha
        return p_init, P_init

    if P_k_1 is None or p_hat_k_1 is None:
        p_hat_k_1, P_k_1 = rec_least_squares(n_a, n_b, k - 1, y, u)

    s_k = np.array(calc_s_k(y, u, n_a, n_b, k)).reshape(-1, 1)
    k_k = (np.dot(P_k_1, s_k)) / (1 + np.dot(s_k.T, np.dot(P_k_1, s_k)))
    P_K = P_k_1 - np.dot(k_k, np.dot(s_k.T, P_k_1))
    p_hat_k = p_hat_k_1 + k_k * (y[k] - np.dot(s_k.T, p_hat_k_1))

    return p_hat_k, P_K
