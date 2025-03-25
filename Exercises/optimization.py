import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from utils import utils
from utils import dbm_math

data1 = utils.read_mat("/home/matthias/WS_all/dbm/raw/ex3_handout/ex3_system1_data.mat")
data2 = utils.read_mat("/home/matthias/WS_all/dbm/raw/ex3_handout/ex3_system2_data.mat")


# Data
t = jnp.array(data2.t1)
y = jnp.array(data2.y1)
u = jnp.array(data2.u1)
Ts = t[1] - t[0]

na_range = jnp.array([3, 6])
nb_range = jnp.array([3, 6])
order_ = jnp.array([2, 7])
num_polys = jnp.array([1, 3])
scalers_poly = jnp.array([0.2, 0.8])


p_hat, loss, func, order, na, nb = dbm_math.hammerstein_optimization(
    y, u, (4, 7), (4, 7), (1, 4), (4, 9), (0.1, 0.9)
)

with open("result.txt", "w") as f:
    f.write(f"Function: {func}\n")
    f.write(f"p_hat: {p_hat}\n")
    f.write(f"loss: {loss}\n")
    f.write(f"order: {order}\n")
    f.write(f"na: {na}\n")
    f.write(f"nb: {nb}\n")
