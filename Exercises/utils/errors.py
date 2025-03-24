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
def mean_absolute_error(y_true: jax.Array, y_pred: jax.Array) -> jax.Array:
    """Mean Absolute Error

    Args:
        y_true (jax.Array): Ground truth values
        y_pred (jax.Array): Predicted values

    Returns:
        jax.Array: Mean Absolute Error
    """
    return jnp.mean(jnp.abs(y_true - y_pred))
