from torchdiffeq import odeint
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Module to describe original ODE
class OCPModule(nn.Module):
    def __init__(self):
        super(OCPModule, self).__init__()
        self.omega = 4.
        self.M1 = torch.diag(torch.tensor([3. * self.omega**2, 0., -self.omega**2], device=device))
        self.M2 = torch.diag(torch.tensor([-2. * self.omega, 0], device=device), -1) + \
                  torch.diag(torch.tensor([2. * self.omega, 0], device=device), 1)
        self.control_net = nn.Sequential(nn.Linear(6, 50), 
                                         nn.Tanh(), 
                                         nn.Linear(50, 3))
        
    # Forward pass: ODE function
    # u = theta1 * y[0:3] + theta2 * y[4:6]
    def forward(self, t, y):
        dydt = torch.zeros_like(y, device=device)
        u = self.control_net(y)
        dydt[0:3] = y[3:6]
        dydt[3:6] = self.M1@y[0:3] + self.M2@y[3:6] + u
        return dydt


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
    t = torch.linspace(0., 0.25, 1000, device=device)
    idx = 0
    for i in range(10000):
        idx += 1
        optimizer.zero_grad()
        y = odeint(ocp_module, y0, t, method='dopri5', rtol=1e-6, atol=1e-6)
        rnorm = torch.norm(y[:, 0:3], dim=1)
        loss = torch.sqrt(torch.sum((y[-1, 0:6] - yf)**2)) + torch.mean(rnorm) - rho
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            print("Iter {:04d} | Loss: {}".format(idx, loss.item()))
        

    # Draw figures
    u = ocp_module.control_net(y)
    u_norm = torch.norm(u, dim=1)
    r_norm = torch.norm(y[:, 0:3], dim=1)
    plt.figure()
    plt.plot(t, r_norm)
    plt.show()

    plt.figure()
    plt.plot(t, u_norm)
    plt.show()

