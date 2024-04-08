import jax
import jax.numpy as jnp
import sympy as sy
from itertools import combinations_with_replacement

'''
File for making feature matrices
'''
# def evaluate_func_list(funcs, vars)

def multivariate_polynomials(state_dim, order):
    def multiply_tuple(tuple):
        val = tuple[0]
        for i in range(1, len(tuple)):
            val *= tuple[i]
        return val

    # Create List of Variables
    vars = []
    for i in range(state_dim):
        vars.append(sy.Symbol(f'x{i}'))
    
    polynomials = []
    for i in range(1, order+1):
        func_tuples = combinations_with_replacement(vars, i)
        for term in func_tuples:
            func = sy.lambdify((vars,), multiply_tuple(term))
            polynomials.append(func)
    return polynomials
    
def sinusoids(state_dim, freq_range=[(0.01)*2*jnp.pi, (1)*2*jnp.pi], resolution=10):
    # Create List of Variables
    vars = []
    for i in range(state_dim):
        vars.append(sy.Symbol(f'x{i}'))
    
    # Get Frequencies
    freqs = jnp.linspace(freq_range[0], freq_range[1], resolution)
    funcs = []
    for var, omega in zip(vars, freqs):
        sinf = jnp.sin(omega*var)
        cosf = jnp.cos(omega*var)
        funcs.extend([sy.lambdify((vars,), sinf), sy.lambdify((vars,), cosf)])
    return funcs

def make_feature_matrix(func_list, include1=True):
    if include1:
        func = lambda x: jnp.array([1.]+[f(x) for f in func_list])
    else:
        func = lambda x: jnp.array([1.]+[f(x) for f in func_list])
        # features = jax.vmap(lambda x: jnp.array([f(x) for f in feature_list]), in_axes=0)
    features = jax.vmap(func, in_axes=0)
    return features

def polynomial_features(state_dim, order):
    polys = multivariate_polynomials(state_dim, order)
    features = make_feature_matrix(polys)
    return features

def main():
    polys = multivariate_polynomials(2, 2)
    # sinusoidal_funcs = sinusoids(state_dim=2, resolution=3)

    features = make_feature_matrix(polys, include1=True)

    data = jnp.ones((3, 2))*2.01
    Phi = features(data)

    print(Phi)

if __name__ == "__main__":
    main()

