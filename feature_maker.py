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
    
def main():
    polys = multivariate_polynomials(2, 3)
    new_func = jax.vmap(lambda x: jnp.array([poly(x) for poly in polys]))
    # print(polys)
    res = new_func(jnp.array([[2, 1],
                        [3, 1]]))
    print(res)
    

if __name__ == "__main__":
    main()

