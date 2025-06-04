import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

def read_trajectory(file_path):
    trajectory = []
    orientations = []
    timestamps = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            data = line.strip().split()
            if len(data) >= 8:
                timestamps.append(float(data[0]))
                position = [float(data[1]), float(data[2]), float(data[3])]
                # ORB-SLAM3 uses [qx, qy, qz, qw] → compatible with scipy
                quaternion = [float(data[4]), float(data[5]), float(data[6]), float(data[7])]
                trajectory.append(position)
                orientations.append(quaternion)
    return np.array(timestamps), np.array(trajectory).T, np.array(orientations)

def average_quaternions(quats):
    q0 = quats[0]
    for i in range(len(quats)):
        if np.dot(q0, quats[i]) < 0:
            quats[i] = -quats[i]
    A = np.zeros((4, 4))
    for q in quats:
        q = q[:, np.newaxis]
        A += q @ q.T
    A /= len(quats)
    eigvals, eigvecs = np.linalg.eigh(A)
    avg_quat = eigvecs[:, np.argmax(eigvals)]
    print("Average quaternion:", avg_quat)
    return avg_quat

def draw_orientation(ax, position, quat, scale=0.1):
    """
    Отрисовывает локальные оси координат, повёрнутые кватернионом.
    Красный - X, зелёный - Y, синий - Z
    """
    r = R.from_quat(quat)

    # Локальные оси
    x_axis = r.apply([scale, 0, 0])
    y_axis = r.apply([0, scale, 0])
    z_axis = r.apply([0, 0, scale])

    x, y, z = position

    ax.quiver(x, y, z, x_axis[0], x_axis[1], x_axis[2], color='red',   length=scale, normalize=True)
    ax.quiver(x, y, z, y_axis[0], y_axis[1], y_axis[2], color='green', length=scale, normalize=True)
    ax.quiver(x, y, z, z_axis[0], z_axis[1], z_axis[2], color='blue',  length=scale, normalize=True)


def plot_traj(ax, stamps, xyz, style, color, label, quats=None, step=1):
    # ax.plot(xyz[0], xyz[1], xyz[2], style, color=color, label=label)
    for i in range(len(stamps)):
        ax.text(xyz[0, i], xyz[1, i], xyz[2, i], str(i), color=color, fontsize=8)
    ax.scatter(xyz[0, 0], xyz[1, 0], xyz[2, 0], color='green', s=100, label='start')
    ax.scatter(xyz[0, -1], xyz[1, -1], xyz[2, -1], color='red', s=100, label='end')

    if quats is not None:
        print("Euler angles [Z,Y,X] in degrees for all poses:")
        counter = 1
        for i, (quat, t) in enumerate(zip(quats, stamps)):
            r = R.from_quat(quat)
            euler = r.as_euler('zyx', degrees=True)
            print(f"{counter:3d}, {t:.6f}: {euler}")
            counter += 1
        # avg_q = average_quaternions(quats.copy())
        # r_avg = R.from_quat(avg_q)
        # direction_avg = r_avg.apply([0, 0, 0.1])

        # В первой точке — средний кватернион (зелёная стрелка)
        
        mid_idx = len(stamps) // 2
        avg_q = average_quaternions(quats.copy())
        draw_orientation(ax, xyz[:, mid_idx], avg_q, scale=0.2)
        
        # ax.quiver(xyz[0, 0], xyz[1, 0], xyz[2, 0],
        #           direction_avg[0], direction_avg[1], direction_avg[2],
        #           color='lime', length=0.2, normalize=True, label='avg quat')
        #             # Получаем углы Эйлера (в радианах) в порядке ZYX

        
        # Во всех остальных — обычные кватернионы (оранжевые стрелки)
        for i in range(step, len(stamps), step):  # пропускаем первую
            quat = quats[i]
            r = R.from_quat(quat)
            direction = r.apply([0, 0, 0.1])
            if i == 1 or i == 58:
                draw_orientation(ax, xyz[:, i], quats[i], scale=0.1)
                # ax.quiver(xyz[0, i], xyz[1, i], xyz[2, i],
                #         direction[0], direction[1], direction[2],
                #         color='orange', length=0.1, normalize=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a trajectory with orientations.")
    parser.add_argument('trajectory_file', help='trajectory file (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument('--plot', help='save plot to a file (e.g. output.pdf)')
    args = parser.parse_args()

    stamps, traj, quats = read_trajectory(args.trajectory_file)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_traj(ax, stamps, traj, '-', 'blue', 'trajectory', quats)

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.legend()

    if args.plot:
        plt.savefig(args.plot)

    plt.gcf().set_size_inches(10, 6)
    plt.tight_layout()
    plt.show()
