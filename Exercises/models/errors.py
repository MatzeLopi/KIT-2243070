import jax
import jax.numpy as jnp


@jax.jit
def mean_squared_error(y_true: jax.Array, y_pred: jax.Array) -> jax.Array:
    """Mean Squared Error

    Args:
        y_true (jax.Array): Ground truth values
        y_pred (jax.Array): Predicted values

    Returns:
        jax.Array: Mean Squared Error
    """
    return jnp.mean((y_true - y_pred) ** 2)


@jax.jit
def _mean_absolute_error(y_true: jax.Array, y_pred: jax.Array) -> jax.Array:
    """Mean Absolute Error

    Args:
        y_true (jax.Array): Ground truth values
        y_pred (jax.Array): Predicted values

    Returns:
        jax.Array: Mean Absolute Error
    """
    return jnp.mean(jnp.abs(y_true - y_pred))


@jax.jit
def mean_absolute_percent_error(y_true: jax.Array, y_pred: jax.Array) -> jax.Array:
    """Mean Absolute Percent Error

    Args:
        y_true (jax.Array): Ground truth values
        y_pred (jax.Array): Predicted values

    Returns:
        jax.Array: Mean Absolute Percent Error
    """
    epsilon = 1e-8
    return jnp.mean(jnp.abs((y_true - y_pred) / (y_true + epsilon))) * 100


@jax.jit
def absolute_error(y_true: jax.Array, y_pred: jax.Array) -> jax.Array:
    """Absolute Error

    Args:
        y_true (jax.Array): Ground truth values
        y_pred (jax.Array): Predicted values

    Returns:
        jax.Array: Absolute Error
    """
    return jnp.abs(y_true - y_pred)


@jax.jit
def absolute_percent_error(y_true: jax.Array, y_pred: jax.Array) -> jax.Array:
    """Absolute Percent Error
    Args:
        y_true (jax.Array): Ground truth values
        y_pred (jax.Array): Predicted values

    Returns:
        jax.Array: Absolute Percent Error
    """
    epsilon = 1e-8
    return jnp.abs((y_true - y_pred) / (y_true + epsilon)) * 100
