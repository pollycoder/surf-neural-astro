import matplotlib.pyplot as plt
import numpy as np





if __name__ == '__main__':
    # Load data
    state_pontryagin = np.load('result/state_pontryagin.npy')
    control_pontryagin = np.load('result/control_pontryagin.npy')
    state_nn = np.load('result/state_nn.npy')
    control_nn = np.load('result/control_nn.npy')
    t_pont = np.linspace(0., 0.25, 1000)
    t_nn = np.linspace(0., 0.25, 100)

    # Plot state
    plt.figure()
    plt.plot(t_pont, state_pontryagin[0, :], label='r1 - Pontryagin')
    plt.plot(t_nn, state_nn[0, :], label='r1 - NN')
    plt.plot(t_pont, state_pontryagin[1, :], label='r2 - Pontryagin')
    plt.plot(t_nn, state_nn[1, :], label='r2 - NN')
    plt.plot(t_pont, state_pontryagin[2, :], label='r3 - Pontryagin')
    plt.plot(t_nn, state_nn[2, :], label='r3 - NN')
    plt.xlabel('t')
    plt.ylabel('r')
    plt.legend()
    plt.show()

    # Plot control
    plt.figure()
    plt.plot(t_pont, control_pontryagin[0, :], label='u1 - Pontryagin')
    plt.plot(t_nn, control_nn[0, :], label='u1 - NN')
    plt.plot(t_pont, control_pontryagin[1, :], label='u2 - Pontryagin')
    plt.plot(t_nn, control_nn[1, :], label='u2 - NN')
    plt.plot(t_pont, control_pontryagin[2, :], label='u3 - Pontryagin')
    plt.plot(t_nn, control_nn[2, :], label='u3 - NN')
    plt.xlabel('t')
    plt.ylabel('u')
    plt.legend()
    plt.show()