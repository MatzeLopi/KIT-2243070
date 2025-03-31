import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from models import hammerstein, utils

data1 = utils.read_mat(r'C:\Users\matze\workspaces\KIT-2243070\raw\ex1_data.mat')
data2 = utils.read_mat(r'C:\Users\matze\workspaces\KIT-2243070\raw\ex2_data.mat')


data1.y = data1.y.flatten()
data1.u = data1.u.flatten()
data1.t = data1.t.flatten()

# Data
t = jnp.array(data2.t1)
y = jnp.array(data2.y1)
u = jnp.array(data2.u1)

na_range = jnp.array([3, 6])
nb_range = jnp.array([3, 6])
order_ = jnp.array([2, 7])
num_polys = jnp.array([1, 3])
scalers_poly = jnp.array([0.2, 0.8])


hp = hammerstein.optimize(data1.y,data1.u,(7,8), (7,8), (1,2), (1,2), (0.1,0.9))

with open("result.txt", "w") as f:
    f.write(f"Best parameters:\n {hp}\n")
