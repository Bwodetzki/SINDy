import numpy as np
import matplotlib.pyplot as plt
import scipy
import jax
import jax.numpy as jnp
from jax import vmap, jit
from jax.lax import scan
import functools as ft

def logm_iterable(vars, n):
  r = vars[0]
  x = vars[1]
  return (r, r*x*(1 - x)), x

@ft.partial(jit, static_argnums=(2,))
@ft.partial(vmap, in_axes=(0, None, None), out_axes=0)
@ft.partial(vmap, in_axes=(None, 0, None), out_axes=0)
def logistic_map(r, x0, n):
  return scan(logm_iterable, (r, x0), jnp.arange(n))[1]


x0s = jnp.linspace(0, 1, 100)
rs = jnp.linspace(0, 4, 100)
n = 1000
data = logistic_map(rs, x0s, n)
# print('hello')

fig, ax = plt.subplots()
ax.plot(rs,
        data.reshape(data.shape[0], data.shape[1] * data.shape[2]),
        '.k',
        ms=0.01,
        alpha=1
)
plt.show()