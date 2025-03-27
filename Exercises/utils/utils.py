# Utils for Data based modeling and control
from scipy import io
import jax.numpy as jnp


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def read_mat(source: str) -> dict:
    """Read a .mat file and return a dictionary with the data

    Args:
        source: path to the .mat file

    Returns:
        data: dictionary with the data
    """
    data = io.loadmat(source)
    data = {
        k: jnp.array(v.flatten()) for k, v in data.items() if not k.startswith("__")
    }
    return dotdict(data)
