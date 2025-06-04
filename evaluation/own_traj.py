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
            if len(data) >= 8:  # tx, ty, tz, qx, qy, qz, qw
                timestamps.append(float(data[0]))
                position = [float(data[1]), float(data[2]), float(data[3])]
                quaternion = [float(data[4]), float(data[5]), float(data[6]), float(data[7])]
                trajectory.append(position)
                orientations.append(quaternion)
    return np.array(timestamps), np.array(trajectory).T, np.array(orientations)

def plot_traj(ax, stamps, xyz, style, color, label, quats=None, step=3):
    ax.plot(xyz[0], xyz[1], xyz[2], style, color=color, label=label)
    ax.scatter(xyz[0, 0], xyz[1, 0], xyz[2, 0], color='green', s=100, label='start')
    ax.scatter(xyz[0, -1], xyz[1, -1], xyz[2, -1], color='red', s=100, label='end')

    if quats is not None:
        for i in range(0, len(stamps), step):
            r = R.from_quat(quats[i])  # quaternion: [x, y, z, w]
            direction = r.apply([0, 0, 0.1])  # ось Z (вперёд) камеры
            ax.quiver(xyz[0, i], xyz[1, i], xyz[2, i],
                      direction[0], direction[1], direction[2],
                      color='orange', length=0.1, normalize=True)

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
    plt.show()
