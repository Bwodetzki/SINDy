import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import data_generator as dg
import feature_maker as fm
from feature_maker import polynomial_features
from tqdm import tqdm

# def feature_generator(num_states=1, polynomials)
'''
Proximal Operators
'''
def l1_prox_op(z, prox_w):
    shrink = jnp.maximum(jnp.abs(z) - prox_w, 0)
    return jnp.sign(z) * shrink, shrink>0

def l0_prox_op(z, prox_w):
    smallinds = jnp.abs(z) < prox_w
    z = z.at[smallinds].set(0)
    return z, smallinds

'''
Core Proximal Optimization
'''
def proximal_optimization(Phi, b, prox_op, alpha, x_init, eps=1e-8, prox_w=0.1):
    x = x_init
    x_old = x_init + 1.

    i = 0
    max_its = 10_000
    while (jnp.linalg.norm(x - x_old) > eps) and (i<max_its):
        x_old = x
        z = x - alpha*Phi.T@(Phi@x - b)
        # z = jnp.linalg.lstsq(Phi, b)[0]
        x = prox_op(z, alpha*prox_w/2)
        i+=1
        if any(jnp.isnan(x)):
            print('here1')
    if i>=max_its:
        print('here')
    return x

'''
SR3 Optimization, Implemented Acoording To: https://uw-amo.github.io/AMO_Site/software/sr3/
'''
def SR3(Phi, b, C, prox_op, x_init, kappa=1, prox_w=0.1, eps=1e-6):
    x = x_init
    w = jnp.zeros_like(x)
    w_old = w + 1.
    A_mat = jnp.vstack((Phi, jnp.sqrt(kappa)*C))

    i = 0
    max_its = 10_000
    while (jnp.linalg.norm(w - w_old) > eps) and (i<max_its):
        w_old = w
        target = jnp.vstack((b, w))
        x = jnp.array(np.linalg.lstsq(A_mat, target, rcond=None)[0])
        y = C@x
        w, _ = prox_op(y, prox_w)

        i+=1
    if i>=max_its:
        print('here')
    return (w, x)
# Proximal variant of SR3, as introduced in the SR3 paper
def SR3_proximal(Phi, b, C, prox_op, x_init, kappa=1, prox_w=0.1, eps=1e-8):
    pass

'''
Original SINDy Regression
'''
def lstsq_reassignment(Phi, b, z, prox_op, prox_w):
    z, smallinds = prox_op(z, prox_w)
    for idx in range(z.shape[1]):
        biginds = smallinds[:, idx] == 0
        z = z.at[biginds, idx].set(jnp.array(np.linalg.lstsq(Phi[:, biginds], b[:, idx], rcond=None)[0]))
    return z

def vanilla_SINDy(Phi, b, prox_w, prox_op=l0_prox_op, eps=1e-8):
    max_iters=10_000
    x = jnp.array(np.linalg.lstsq(Phi, b, rcond=None)[0])
    x_old = x + 0.1
    i=0
    while jnp.linalg.norm(x - x_old) > eps and i < max_iters:
        x_old = x
        x = lstsq_reassignment(Phi, b, x, prox_op, prox_w)
        i+=0
    return x


'''
Cross Validation for Optimization Algorithms
'''
def cross_validation(optimization_func, features, train_data, test_data, param_ranges, resolution=100):
    '''
    optimization must be a function that takes 3 input parameters
    '''
    solutions = []
    errors = []
    params = jnp.linspace(param_ranges[0], param_ranges[1], resolution)
    for param in tqdm(params):
        sol = optimization_func(features(train_data[0]), train_data[1], param)
        # sol = sol.flatten()
        solutions.append(sol)

        error = jnp.linalg.norm(features(test_data[0])@sol - test_data[1])
        errors.append(error)
    idx = jnp.argmin(jnp.array(errors))

    return (solutions[idx], errors[idx], params[idx])

def proximal_CV(X, y, features, alpha, key, proximal_operator=l1_prox_op, test_size=0.25, param_ranges=[0.001, 10], resolution=1000):
    key, subkey = jax.random.split(key)
    size = len(features(0))
    x_init = jax.random.normal(subkey, (size,1))

    key, subkey = jax.random.split(key)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(subkey[0]))
    opt_fun = lambda phi_mat, data, param: proximal_optimization(phi_mat, data, proximal_operator, alpha, x_init, prox_w=param)
    fit, error, prox_w = cross_validation(opt_fun, features, 
                                            train_data=(X_train, y_train), 
                                            test_data=(X_test, y_test), 
                                            param_ranges=param_ranges, 
                                            resolution=resolution)
    return fit, error, prox_w

def SR3_CV(X, y, features, C, kappa, key, proximal_operator=l1_prox_op, test_size=0.25, param_ranges=[0.001, 10], resolution=1000):
    key, subkey = jax.random.split(key)
    size = features(X).shape[1]
    x_init = jax.random.normal(subkey, (size, X.shape[1]))

    key, subkey = jax.random.split(key)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(subkey[0]))
    opt_fun = lambda phi_mat, data, param: SR3(phi_mat, data, C, proximal_operator, x_init, kappa=kappa, prox_w=param)[0]
    fit, error, prox_w = cross_validation(opt_fun, features, 
                                            train_data=(X_train, y_train), 
                                            test_data=(X_test, y_test), 
                                            param_ranges=param_ranges, 
                                            resolution=resolution)
    return fit, error, prox_w

def SINDy_CV(X, y, features, key, test_size=0.25, param_ranges=[0.0001, 10], resolution=1000, verbose=True):
    key, subkey = jax.random.split(key)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(subkey[0]))
    opt_fun = vanilla_SINDy
    fit, error, prox_w = cross_validation(opt_fun, features, 
                                            train_data=(X_train, y_train), 
                                            test_data=(X_test, y_test), 
                                            param_ranges=param_ranges, 
                                            resolution=resolution)
    if verbose:
        print(f"The optimal proximal weight found is {prox_w}")
    return fit, error, prox_w

def main():
    # key, subkey = jax.random.split(jax.random.PRNGKey(0))
    # fun = lambda x: 3*x**3 - x**2 - 2*x + 3
    # x, y = general_func(fun, subkey, lim=[-2.5, 2.5], num_samples=1000, sigma=1)

    # #    poly = PolynomialFeatures(degree=5)
    # #    Phi = poly.fit_transform(x)
    # size = 5
    # features = lambda x: jnp.array([x**i for i in range(size)]).T
    # Phi = features(x)

    # key, subkey = jax.random.split(key)
    # x_init = jax.random.normal(subkey, (size,))

    # # fit = proximal_optimization(Phi, y, l1_prox_op, 0.00001, x_init, prox_w=3)

    # C = jnp.eye(size)

    # ## Cross Validation
    # key, subkey = jax.random.split(key)
    # fit, error, prox_w = proximal_CV(x, y, features, alpha=0.00001, key=subkey, param_ranges=[0.001, 100], resolution=100)
    # print(error)
    # print(fit)

    # plt.figure()
    # plt.plot(x, y)
    # plt.plot(x, Phi@fit)
    # plt.show()

    # Get data
    t, bigX, dX = dg.solve_lorenz_sys(10, noise_sigma=0)
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Get features
    features, feature_list, feature_name = polynomial_features(state_dim=3, order=3)
    Phi = features(bigX)

    # x_init = jax.random.normal(subkey, (Phi.shape[1], Y.shape[1]))

    # Fit the function
    ##
    # fit = vanilla_SINDy(Phi, dX, prox_w=0.05, prox_op=l0_prox_op)
    ##
    # key, subkey = jax.random.split(jax.random.PRNGKey(0))
    # fit, _, _ = SINDy_CV(bigX, dX, features, subkey)
    ##
    # C = jnp.eye(len(Phi[0]))
    # key, subkey = jax.random.split(jax.random.PRNGKey(0))
    # x_init = jax.random.normal(subkey, (Phi.shape[1], bigX.shape[1]))
    # fit, _ = SR3(Phi, dX, C, l0_prox_op, x_init)
    ##
    C = jnp.eye(len(Phi[0]))
    key, subkey = jax.random.split(jax.random.PRNGKey(0))
    fit, _, _ = SR3_CV(bigX, dX, features, C, 1, subkey, proximal_operator=l1_prox_op, test_size=0.25, param_ranges=[0.001, 10], resolution=100)

    
    # Get integrator
    fm.get_functions(fit, feature_name)
    integrator = fm.get_integrator(fit, feature_list)

    # Integrate
    from scipy.integrate import solve_ivp
    sol = solve_ivp(integrator, [0, 10], [-8.0, 7.0, 27.0], t_eval=np.linspace(0, 10, 1000))
    x, y, z = sol.y

    fig, ax = plt.subplots(1, 2, subplot_kw={"projection": "3d"})

    ax[1].plot(x, y, z)
    ax[0].plot(bigX[:, 0], bigX[:, 1], bigX[:, 2])
    ax[0].set_title("True Dynamical System")
    ax[1].set_title("Predicted Dynamical System")
    fig.suptitle("$l_1$ Regularized SR3")
    plt.show()
    print('here')

if __name__ == "__main__":
    main() 