import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from data_generator import general_func
from tqdm import tqdm

# def feature_generator(num_states=1, polynomials)
'''
Proximal Operators
'''
def l1_prox_op(z, prox_w):
    return jnp.sign(z) * jnp.maximum(abs(z) - prox_w, 0)

def l0_prox_op(z, prox_w):
    z_updated = []
    for i in range(len(z)):
        val = z[i] if z[i]>prox_w else [0.]
        z_updated.append(val)
    return jnp.array(z_updated)

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
        target = jnp.vstack((b.reshape(-1, 1), w.reshape(-1, 1)))
        x = jnp.linalg.lstsq(A_mat, target)[0]
        y = C@x
        w = prox_op(y, prox_w)

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
    smallinds = prox_op(z, prox_w) == 0
    z[smallinds] = 0
    for idx in range(z.shape[1]):
        biginds = smallinds[:, idx] == 0
        z[biginds, idx] = jnp.linalg.lstsq(Phi[:, biginds], b[:, idx])[0]
    return z
    
    

def vanilla_SINDy(Phi, b, x_init, prox_w, eps=1e-8):
    x = x_init
    x_old = x_init+1.

    # Main Loop
    x = jnp.linalg.lstsq(Phi, b)[0]
    while jnp.linalg.norm(x - x_old) > eps:
        x_old = x
        x = lstsq_reassignment(Phi, b, x, l0_prox_op, prox_w)
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
        sol = sol.flatten()
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
    size = len(features(0))
    x_init = jax.random.normal(subkey, (size,))

    key, subkey = jax.random.split(key)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(subkey[0]))
    opt_fun = lambda phi_mat, data, param: SR3(phi_mat, data, C, proximal_operator, x_init, kappa=kappa, prox_w=param)[0]
    fit, error, prox_w = cross_validation(opt_fun, features, 
                                            train_data=(X_train, y_train), 
                                            test_data=(X_test, y_test), 
                                            param_ranges=param_ranges, 
                                            resolution=resolution)
    return fit, error, prox_w

def main():
    key, subkey = jax.random.split(jax.random.PRNGKey(0))
    fun = lambda x: 3*x**3 - x**2 - 2*x + 3
    x, y = general_func(fun, subkey, lim=[-2.5, 2.5], num_samples=1000, sigma=1)

    #    poly = PolynomialFeatures(degree=5)
    #    Phi = poly.fit_transform(x)
    size = 5
    features = lambda x: jnp.array([x**i for i in range(size)]).T
    Phi = features(x)

    key, subkey = jax.random.split(key)
    x_init = jax.random.normal(subkey, (size,))

    # fit = proximal_optimization(Phi, y, l1_prox_op, 0.00001, x_init, prox_w=3)

    C = jnp.eye(size)

    ## Cross Validation
    key, subkey = jax.random.split(key)
    fit, error, prox_w = proximal_CV(x, y, features, alpha=0.00001, key=subkey, param_ranges=[0.001, 100], resolution=100)
    print(error)
    print(fit)

    plt.figure()
    plt.plot(x, y)
    plt.plot(x, Phi@fit)
    plt.show()

if __name__ == "__main__":
    main() 