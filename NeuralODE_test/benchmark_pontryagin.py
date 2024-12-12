import sys
import argparse
sys.path.append('..')

from scipy.integrate import solve_bvp, trapezoid
import numpy as np
import matplotlib.pyplot as plt
from utils.dynamics import ocp_dyn_bvp, ocp_dyn_bc, lagrange_multiplier
from nn_models.ocp_model import OCPModule

'''
Benchmark: Optimal control problem
    Method 1: Solve the optimal control problem using Pontryagin's Minimum
    Method 2: Use MLP to describe the control function
'''


if __name__ == '__main__':
    
    # t and initial guess
    t0 = 0.
    tf = 0.25
    t = np.linspace(t0, tf, 100)
    y0 = np.ones((12, t.size))
    y0[0:6, 0] = np.array([10., 0., 0., 0., 0., np.pi])
    y0[0:6, -1] = np.array([0., 10., 0., 0., 0., np.pi])

    # Constants
    omega = 4.
    rho = 10.
    M1 = np.diag([3. * omega**2, 0., -omega**2])
    M2 = np.diag([-2. * omega, 0], -1) + np.diag([2. * omega, 0], 1)

    # Solve the BVP
    sol = solve_bvp(ocp_dyn_bvp, ocp_dyn_bc, t, y0, bc_tol=1e-6, max_nodes=10000)
    u = -sol.sol(t)[9:12]
    for i in range(0, t.size):
        r = sol.sol(t)[0:3, i]
        v = sol.sol(t)[3:6, i]
        lambda46 = sol.sol(t)[9:12, i]
        mu = lagrange_multiplier(r, v, lambda46, M1, M2, rho)
        u[:, i] = 2 * mu * r - lambda46


    # Save data
    np.save('./result/data/state_pontryagin.npy', sol.sol(t)[0:6])
    np.save('./result/data/control_pontryagin.npy', u)

    # Objective: energy
    u_norm = np.linalg.norm(u, axis=0)
    r_norm = np.linalg.norm(sol.sol(t)[0:3], axis=0)
    J = 0.5 * trapezoid(u_norm * u_norm, t)
    print("J = {}".format(J))

    
