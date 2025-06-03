import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_trajectory(file_path):
    trajectory = []
    timestamps = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            data = line.strip().split()
            if len(data) >= 4:
                timestamps.append(float(data[0]))
                position = [float(data[1]), float(data[2]), float(data[3])]
                trajectory.append(position)
    return np.array(timestamps), np.array(trajectory).T  # возвращаем (3, N)

def plot_traj(ax, stamps, xyz, style, color, label):
    ax.plot(xyz[0], xyz[1], xyz[2], style, color=color, label=label)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a trajectory from a single file.")
    parser.add_argument('trajectory_file', help='trajectory file (format: timestamp tx ty tz ...)')
    parser.add_argument('--plot', help='save plot to a file (e.g. output.pdf)')
    args = parser.parse_args()

    stamps, traj = read_trajectory(args.trajectory_file)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_traj(ax, stamps, traj, '-', 'blue', 'trajectory')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.legend()
    if args.plot:
        plt.savefig(args.plot)
    plt.show()
