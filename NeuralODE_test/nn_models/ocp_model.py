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
            act=nn.LeakyReLU,
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
        J = torch.from_numpy(np.array(J))
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
    duration = 0
    loss_prev = 1.e10
    y_bench = torch.tensor(np.transpose(np.load('../result/data/state_pontryagin.npy')))
    for i in range(10000):
        idx += 1
        optimizer.zero_grad()
        y = odeint(ocp_module, y0, t, method='dopri5', rtol=1e-8, atol=1e-8, options={'dtype': torch.float32})
        rnorm = torch.norm(y[:, 0:3], dim=1)
        loss = torch.norm(y[-1, 0:6] - yf[0:6])**2
        loss.requires_grad_(True)

        if loss.item() < loss_prev:
            loss_prev = loss.item()
            duration = 0
        else:
            duration += 1
            if duration > 500:
                break
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            print("Iter {:05d}\t| Loss: {:05f}\t| Energy: {:05f}\t| Terminal error: {:05f}\t".format(idx, loss.item(), ocp_module.energy_objective(t, y), torch.norm(y[-1, 0:6] - yf[0:6])))
        
    torch.save(ocp_module, '../result/model/ocp_module_tanh.pth')

    # Draw figures
    ocp_model = torch.load('../result/model/ocp_module_tanh.pth')
    ocp_model.eval()
    t_test = torch.linspace(0., 0.25, 1000)
    y_test = odeint(ocp_model, y0, t_test, method='dopri5', rtol=1e-8, atol=1e-8, options={'dtype': torch.float32})
    u = ocp_module.control_net(y_test)
    u_norm = torch.norm(u, dim=1).detach().cpu().numpy()
    r_norm = torch.norm(y_test[:, 0:3], dim=1).detach().cpu().numpy()
    print(r_norm)

    J = ocp_module.energy_objective(t_test, y_test)
    print("Energy Objective: {}".format(J))

    u = u.detach().cpu().numpy()
    y = y_test.detach().cpu().numpy()
    np.save('../result/data/control_mlp.npy', u)
    np.save('../result/data/state_mlp.npy', y)
