import matplotlib.pyplot as plt
import numpy as np





if __name__ == '__main__':
    # Load data
    state_pontryagin = np.load('./result/data/state_pontryagin.npy')
    control_pontryagin = np.load('./result/data/control_pontryagin.npy')
    state_nn = np.load('./result/data/state_mlp.npy')
    control_nn = np.load('./result/data/control_mlp.npy')

    state_nn = np.transpose(state_nn)
    control_nn = np.transpose(control_nn)
    dist_nn = np.linalg.norm(state_nn[0:3, -1])
    dist_pontryagin = np.linalg.norm(state_pontryagin[0:3, -1])
    unorm_nn = np.linalg.norm(control_nn, axis=0)
    unorm_pontryagin = np.linalg.norm(control_pontryagin, axis=0)

    print(state_pontryagin.shape)
    print(control_pontryagin.shape)
    print(state_nn.shape)
    print(control_nn.shape)
    t_pont = np.linspace(0., 0.25, 100)
    t_nn = np.linspace(0., 0.25, 1000)

    # Plot state
    plt.figure()
    plt.plot(t_pont, state_pontryagin[0, :], label='r1 - Pontryagin', color='red')
    plt.plot(t_nn, state_nn[0, :], label='r1 - NN', linestyle='--', color='red')
    plt.plot(t_pont, state_pontryagin[1, :], label='r2 - Pontryagin', color='blue')
    plt.plot(t_nn, state_nn[1, :], label='r2 - NN', linestyle='--', color='blue')
    plt.plot(t_pont, state_pontryagin[2, :], label='r3 - Pontryagin', color='green')
    plt.plot(t_nn, state_nn[2, :], label='r3 - NN', linestyle='--', color='green')
    plt.xlabel('t')
    plt.ylabel('r')
    plt.title('Position r')
    plt.legend()
    plt.savefig('./result/fig/r.png')

    # Plot control
    plt.figure()
    plt.plot(t_pont, control_pontryagin[0, :], label='u1 - Pontryagin', color='red')
    plt.plot(t_nn, control_nn[0, :], label='u1 - NN', linestyle='--', color='red')
    plt.plot(t_pont, control_pontryagin[1, :], label='u2 - Pontryagin', color='blue')
    plt.plot(t_nn, control_nn[1, :], label='u2 - NN', linestyle='--', color='blue')
    plt.plot(t_pont, control_pontryagin[2, :], label='u3 - Pontryagin', color='green')
    plt.plot(t_nn, control_nn[2, :], label='u3 - NN', linestyle='--', color='green')
    plt.xlabel('t')
    plt.ylabel('u')
    plt.title('Control u')
    plt.legend()
    plt.savefig('./result/fig/u.png')

    # Plot distance
    plt.figure()
    plt.plot(t_pont, np.linalg.norm(state_pontryagin[0:3], axis=0), label='Pontryagin', color='red')
    plt.plot(t_nn, np.linalg.norm(state_nn[0:3], axis=0), label='NN', color='blue')
    plt.xlabel('t')
    plt.ylabel('r')
    plt.title('Distance r')
    plt.legend()
    plt.savefig('./result/fig/dist.png')

    # Plot control norm
    plt.figure()
    plt.plot(t_pont, unorm_pontryagin, label='Pontryagin', color='red')
    plt.plot(t_nn, unorm_nn, label='NN', color='blue')
    plt.xlabel('t')
    plt.ylabel('u')
    plt.title('Control norm')
    plt.legend()
    plt.savefig('./result/fig/unorm.png')

    # Plot trajectory (3D)
    # Earth: (0, 0, 0)
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(state_pontryagin[0, :], state_pontryagin[1, :], state_pontryagin[2, :], 'red', label='Pontryagin')
    ax.plot(state_nn[0, :], state_nn[1, :], state_nn[2, :], 'blue', label='NN')
    ax.scatter(0., 0., 0., color='black', label='Earth')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Trajectory')
    ax.axis('equal')
    ax.legend()
    plt.savefig('./result/fig/traj3d.png')
