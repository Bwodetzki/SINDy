import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def general_func(fun, key, lim=[0, 10], num_samples=100, sigma=0):
    x = jnp.linspace(lim[0], lim[1], num_samples)
    y = fun(x)
    corrupt_y = y + jax.random.normal(key, (num_samples,))*sigma

    return (x, corrupt_y)

def solve_lorenz_sys(tf, num_points=1000, sigma=10, rho=28, beta=8/3, ics=[-8.0, 7.0, 27.0]):
    def lorenz_eoms(t, y, params):
        sigma, rho, beta = params
        x1, x2, x3 = y

        x1d = sigma*(x2 - x1)
        x2d = x1*(rho - x3) - x2
        x3d = x1*x2 - beta*x3
        return [x1d, x2d, x3d]

    t_span = [0, tf]
    sample_points = jnp.linspace(0, tf, num_points)

    options = {
        'reltol' : 1e-12,
        'abstol' : 1e-12
    }

    sol = solve_ivp(lorenz_eoms, t_span, ics, t_eval=sample_points, args=((sigma, rho, beta),), **options)
    x, y, z = sol.y
    return (sol.t, x, y, z)


def main():
    ## Solve Lorenz System
    t, x, y, z = solve_lorenz_sys(10)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    ax.plot(x, y, z)
    plt.show()

    ## Solve General Function
    key, subkey = jax.random.PRNGKey(0)
    fun = lambda x: 3*x**3 - x**2 - 2*x + 3
    x, y = general_func(fun, subkey, lim=[-2.5, 2.5], sigma=0.5)

    plt.plot(x, y)
    plt.show()

if __name__ == "__main__":
    main()