import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from data_generator import general_func

# def feature_generator(num_states=1, polynomials)

def l1_prox_op(z, prox_w):
    #  return jnp.maximum(0.,z-prox_w/2)-jnp.maximum(0.,-z-prox_w/2)
    return jnp.sign(z) * jnp.maximum(abs(z) - prox_w, 0)

def proximal_optimization(Phi, b, prox_op, alpha, x_init, eps=1e-8, prox_w=0.1):
    x = x_init
    x_old = x_init + 1.

    i = 0
    max_its = 10_000
    while (jnp.linalg.norm(x - x_old) > eps) and (i<max_its):
        x_old = x
        # z = x - alpha*Phi.T@(Phi@x - b)
        z = jnp.linalg.lstsq(Phi, b)[0]
        x = prox_op(z, alpha, alpha*prox_w/2)
        i+=1
    if i>=max_its:
        print('here')
    return x

def SR3(Phi, b, C, prox_op, x_init, kappa=1, prox_w=0.1, eps=1e-8):
    x = x_init
    w = jnp.zeros_like(x)
    w_old = w + 1.
    A_mat = jnp.vstack((Phi, jnp.sqrt(kappa)*C))

    i = 0
    max_its = 10_000
    while (jnp.linalg.norm(w - w_old) > eps) and (i<max_its):
        w_old = w
        target = jnp.vstack((b.reshape(-1, 1), w.reshape(-1, 1)))
        x = jnp.linalg.lstsq(A_mat, target)[0]
        y = C@x
        w = prox_op(y, prox_w)

        i+=1
    if i>=max_its:
        print('here')
    return (w, x)

def SR3_proximal(Phi, b, C, prox_op, x_init, kappa=1, prox_w=0.1, eps=1e-8):
    pass


def cross_validation(optimization_func, features, train_data, test_data, param_ranges, resolution=100):
    '''
    optimization must be a function that takes 3 input parameters
    '''
    solutions = []
    errors = []
    params = jnp.linspace(param_ranges[0], param_ranges[1], resolution)
    for param in params:
        sol = optimization_func(features(train_data[0]), train_data[1], param)
        sol = sol.flatten()
        solutions.append(sol)

        error = jnp.linalg.norm(features(test_data[0])@sol - test_data[1])
        errors.append(error)
    idx = jnp.argmin(jnp.array(errors))

    return (solutions[idx], errors[idx], params[idx])

def main():
   key, subkey = jax.random.split(jax.random.PRNGKey(0))
   fun = lambda x: 3*x**3 - x**2 - 2*x + 3
   x, y = general_func(fun, subkey, lim=[-2.5, 2.5], num_samples=1000, sigma=1)

#    poly = PolynomialFeatures(degree=5)
#    Phi = poly.fit_transform(x)
   size = 8
   features = lambda x: jnp.array([x**i for i in range(size)]).T
   Phi = features(x)

   key, subkey = jax.random.split(key)
   x_init = jax.random.normal(subkey, (size,))

#    fit = proximal_optimization(Phi, y, l1_prox_op, 0.0001, x_init, prox_w=0.3)
   
   C = jnp.eye(size)
   fit, _ = SR3(Phi, y, C, l1_prox_op, x_init, kappa=1, prox_w=0.1)
#    print(fit)

   ## Cross Validation
   key, subkey = jax.random.split(key)
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=int(subkey[0]))
   opt_fun = lambda phi_mat, data, param: SR3(phi_mat, data, C, l1_prox_op, x_init, kappa=1, prox_w=param)[0]
   fit, error, prox_w = cross_validation(opt_fun, features, 
                                         train_data=(x_train, y_train), 
                                         test_data=(x_test, y_test), 
                                         param_ranges=[0.001, 10], 
                                         resolution=1000)
   print(error)
   print(fit)

#    fit = jnp.linalg.lstsq(Phi, y)[0]

   plt.figure()
   plt.plot(x, y)
   plt.plot(x, Phi@fit)
   plt.show()

if __name__ == "__main__":
    main()