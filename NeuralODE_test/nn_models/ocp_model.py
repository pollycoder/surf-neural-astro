from torchdiffeq import odeint
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

device = torch.device('cpu')

# Loss: Sine
class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)

# Module to describe original ODE
class OCPModule(nn.Module):
    def __init__(self):
        super(OCPModule, self).__init__()
        self.omega = 4.
        self.M1 = torch.diag(torch.tensor([3. * self.omega**2, 0., -self.omega**2], device=device))
        self.M2 = torch.diag(torch.tensor([-2. * self.omega, 0], device=device), -1) + \
                  torch.diag(torch.tensor([2. * self.omega, 0], device=device), 1)
        self.control_net = mlp(
            input_dim=6,
            hidden_dim=64,
            output_dim=3,
            hidden_depth=2,
            act=nn.ReLU,
        )
        
    # Forward pass: ODE function
    # u = theta1 * y[0:3] + theta2 * y[4:6]
    def forward(self, t, y):
        dydt = torch.zeros_like(y, device=device)
        u = self.control_net(y)
        dydt[0:3] = y[3:6]
        dydt[3:6] = self.M1@y[0:3] + self.M2@y[3:6] + u
        return dydt
    
    def energy_objective(self, t, y):
        u = self.control_net(y)
        u_norm = torch.norm(u, dim=1).detach().cpu().numpy()
        J = 0.5 * integrate.trapezoid(u_norm**2, t.detach().cpu().numpy())
        return J



def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None, act=nn.ReLU):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), act()]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), act()]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk



# Solve the ODE
if __name__ == '__main__':
    # Boundary conditions
    y0 = torch.tensor([10., 0., 0., 0., 0., torch.pi], device=device)
    yf = torch.tensor([0., 10., 0., 0., 0., torch.pi], device=device)
    rho = torch.tensor(10., device=device)

    # ODE Module
    ocp_module = OCPModule().to(device)
    optimizer = optim.Adam(ocp_module.parameters(), lr=1e-3)

    # Train ODE Module
    t = torch.linspace(0., 0.25, 100, device=device)
    idx = 0
    for i in range(10000):
        idx += 1
        optimizer.zero_grad()
        y = odeint(ocp_module, y0, t, method='dopri5', rtol=1e-8, atol=1e-8)
        rnorm = torch.norm(y[:, 0:3], dim=1)
        loss = torch.mean(torch.abs(rnorm - rho)) + torch.norm(y[-1, :] - yf)
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            print("Iter {:04d} | Loss: {}".format(idx, loss.item()))
        

    # Draw figures
    u = ocp_module.control_net(y)
    u_norm = torch.norm(u, dim=1).detach().cpu().numpy()
    r_norm = torch.norm(y[:, 0:3], dim=1).detach().cpu().numpy()
    np.save('../result/control_mlp.npy', u)
    np.save('../result/state_mlp.npy', y)
    
    J = ocp_module.energy_objective(t, y)
    print("Energy Objective: {}".format(J))
    plt.figure()
    plt.plot(t, r_norm)
    plt.show()
    plt.savefig('../result/r_norm.png')

    plt.figure()
    plt.plot(t, u_norm)
    plt.show()
    plt.savefig('../result/u_norm.png')
