import numpy as np


# Real ODE Function for OCP - Pontryagin's Minimum Principle
# u = -lambda46 + 2 * mu * r
def lagrange_multiplier(r, v, lambda46, M1, M2, rho):
    mu = (np.dot(r, lambda46) - np.dot(v, v) - np.transpose(r)@M1@r - np.transpose(r)@M2@v) / (2. * rho**2)
    return mu

def ocp_dyn_bvp(t, y):
    dydt = np.zeros_like(y)
    omega = 4.
    rho = 10.
    M1 = np.diag([3. * omega**2, 0., -omega**2])
    M2 = np.diag([-2. * omega, 0], -1) + np.diag([2. * omega, 0], 1)
    u = np.zeros((3, y.shape[1]))

    for i in range(3, y.shape[1]):
        r = y[0:3, i]
        v = y[3:6, i]
        lambda13 = y[6:9, i]
        lambda46 = y[9:12, i]
        mu = 0.#lagrange_multiplier(r, v, lambda46, M1, M2, rho)
        u[:, i] = 2 * mu * r - lambda46

        dydt[0:3, i] = v
        dydt[3:6, i] = M1@r + M2@v + u[:, i]
        dydt[6:9, i] = -M1@lambda46 + 4. * mu * M1@r - 2. * mu * M2@v - 4. * mu**2 * r + 2. * mu * lambda46
        dydt[9:12, i] = -lambda13 + M2@lambda46 + 4. * mu * v - 2. * mu * M2@r

    
    dydt = np.vstack((dydt[0], dydt[1], dydt[2], dydt[3], dydt[4], dydt[5], dydt[6], dydt[7], dydt[8], dydt[9], dydt[10], dydt[11]))
    return dydt

# Boundary conditions for OCP
# y0 = [10., 0., 0., 0., 0., np.pi]
# yf = [0., 10., 0., 0., 0., np.pi]
def ocp_dyn_bc(y0, yf):
    y0c = np.array([10., 0., 0., 0., 0., np.pi])
    yfc = np.array([0., 10., 0., 0., 0., np.pi])
    return np.concatenate((y0[0:6] - y0c, yf[0:6] - yfc))