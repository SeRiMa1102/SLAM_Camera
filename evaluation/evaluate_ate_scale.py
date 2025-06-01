# Modified by Raul Mur-Artal
# Automatically compute the optimal scale factor for monocular VO/SLAM.

# Software License Agreement (BSD License)
#
# Copyright (c) 2013, Juergen Sturm, TUM
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of TUM nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Requirements: 
# sudo apt-get install python-argparse

"""
This script computes the absolute trajectory error from the ground truth
trajectory and the estimated trajectory.
"""

import sys
import numpy
import argparse
import associate

import numpy as np

def align(model, data):
    """
    Align two trajectories using the method of Horn (closed-form).

    Args:
        model (np.ndarray): estimated trajectory, shape (3, N)
        data (np.ndarray): ground truth trajectory, shape (3, N)

    Returns:
        rot (np.ndarray): rotation matrix (3x3)
        transGT (np.ndarray): translation vector (3x1), with scale
        trans_errorGT (np.ndarray): translation error (with scale)
        trans (np.ndarray): translation vector (3x1), without scale
        trans_error (np.ndarray): translation error (no scale)
        s (float): scale factor
    """
    np.set_printoptions(precision=3, suppress=True)

    # Центрирование траекторий
    model_mean = model.mean(axis=1, keepdims=True)
    data_mean = data.mean(axis=1, keepdims=True)
    model_zerocentered = model - model_mean
    data_zerocentered = data - data_mean

    # Ковариационная матрица
    W = model_zerocentered @ data_zerocentered.T
    U, _, Vt = np.linalg.svd(W.T)

    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1

    rot = U @ S @ Vt

    # Масштаб
    rotmodel = rot @ model_zerocentered
    dots = np.sum(data_zerocentered * rotmodel)
    norms = np.sum(model_zerocentered ** 2)
    s = float(dots / norms)

    # Трансляции
    transGT = data_mean - s * rot @ model_mean
    trans = data_mean - rot @ model_mean

    # Выравнивание
    model_alignedGT = s * (rot @ model) + transGT
    model_aligned = rot @ model + trans

    # Ошибки
    alignment_errorGT = model_alignedGT - data
    alignment_error = model_aligned - data
    trans_errorGT = numpy.linalg.norm(alignment_errorGT, axis=0)
    trans_error = numpy.linalg.norm(alignment_error, axis=0)

    return rot, transGT, trans_errorGT, trans, trans_error, s

def plot_traj(ax,stamps,traj,style,color,label):
    """
    Plot a trajectory using matplotlib. 
    
    Input:
    ax -- the plot
    stamps -- time stamps (1xn)
    traj -- trajectory (3xn)
    style -- line style
    color -- line color
    label -- plot legend
    
    """
    stamps.sort()
    interval = numpy.median([s-t for s,t in zip(stamps[1:],stamps[:-1])])
    x = []
    y = []
    last = stamps[0]
    for i in range(len(stamps)):
        if stamps[i]-last < 2*interval:
            x.append(traj[i][0])
            y.append(traj[i][1])
        elif len(x)>0:
            ax.plot(x,y,style,color=color,label=label)
            label=""
            x=[]
            y=[]
        last= stamps[i]
    if len(x)>0:
        ax.plot(x,y,style,color=color,label=label)
            
def plot_traj_3d(ax, stamps, xyz, style, color, label):
    ax.plot(xyz[0], xyz[1], xyz[2], style, color=color, label=label)

if __name__=="__main__":
    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script computes the absolute trajectory error from the ground truth trajectory and the estimated trajectory. 
    ''')
    parser.add_argument('first_file', help='ground truth trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument('second_file', help='estimated trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument('--offset', help='time offset added to the timestamps of the second file (default: 0.0)',default=0.0)
    parser.add_argument('--scale', help='scaling factor for the second trajectory (default: 1.0)',default=1.0)
    parser.add_argument('--max_difference', help='maximally allowed time difference for matching entries (default: 10000000 ns)',default=20000000)
    parser.add_argument('--save', help='save aligned second trajectory to disk (format: stamp2 x2 y2 z2)')
    parser.add_argument('--save_associations', help='save associated first and aligned second trajectory to disk (format: stamp1 x1 y1 z1 stamp2 x2 y2 z2)')
    parser.add_argument('--plot', help='plot the first and the aligned second trajectory to an image (format: png)')
    parser.add_argument('--verbose', help='print all evaluation data (otherwise, only the RMSE absolute translational error in meters after alignment will be printed)', action='store_true')
    parser.add_argument('--verbose2', help='print scale eror and RMSE absolute translational error in meters after alignment with and without scale correction', action='store_true')
    args = parser.parse_args()

    first_list = associate.read_file_list(args.first_file, False)
    second_list = associate.read_file_list(args.second_file, False)

    matches = associate.associate(first_list, second_list,float(args.offset),float(args.max_difference))    
    if len(matches)<2:
        sys.exit("Couldn't find matching timestamp pairs between groundtruth and estimated trajectory! Did you choose the correct sequence?")
    # Преобразование матчей в массивы (3, N)
    first_xyz = np.array([[float(value) for value in first_list[a][0:3]] for a, b in matches]).T
    second_xyz = np.array([[float(value) * float(args.scale) for value in second_list[b][0:3]] for a, b in matches]).T

    # Отсортированный список
    sorted_second_list = sorted(second_list.items())
    second_xyz_full = np.array([[float(value) * float(args.scale) for value in values[0:3]]
                                for _, values in sorted_second_list]).T

    # Выравнивание
    rot, transGT, trans_errorGT, trans, trans_error, scale = align(second_xyz, first_xyz)
    
    # Трансформация всех траекторий
    second_xyz_aligned = scale * (rot @ second_xyz) + trans
    second_xyz_notscaled = rot @ second_xyz + trans
    second_xyz_notscaled_full = rot @ second_xyz_full + trans

    # Подготовка полных координат (3, N)
    first_stamps = sorted(first_list.keys())
    first_xyz_full = np.array([[float(value) for value in first_list[b][0:3]] for b in first_stamps]).T

    second_stamps = sorted(second_list.keys())
    second_xyz_full = np.array([[float(value) * float(args.scale) for value in second_list[b][0:3]] for b in second_stamps]).T
    second_xyz_full_aligned = scale * (rot @ second_xyz_full) + transGT + np.array([[1], [1], [0]])

    
    if args.verbose:
        print("compared_pose_pairs %d pairs"%(len(trans_error)))

        print("absolute_translational_error.rmse %f m"%numpy.sqrt(numpy.dot(trans_error,trans_error) / len(trans_error)))
        print("absolute_translational_error.mean %f m"%numpy.mean(trans_error))
        print("absolute_translational_error.median %f m"%numpy.median(trans_error))
        print("absolute_translational_error.std %f m"%numpy.std(trans_error))
        print("absolute_translational_error.min %f m"%numpy.min(trans_error))
        print("absolute_translational_error.max %f m"%numpy.max(trans_error))
        print("max idx: %i" %numpy.argmax(trans_error))
    else:
        print("%f, %f " % (numpy.sqrt(numpy.dot(trans_error,trans_error) / len(trans_error)),  scale))
        print("%f,%f" % (numpy.sqrt(numpy.dot(trans_error,trans_error) / len(trans_error)),  scale))
        print("%f,%f,%f" % (numpy.sqrt(numpy.dot(trans_error,trans_error) / len(trans_error)), scale, numpy.sqrt(numpy.dot(trans_errorGT,trans_errorGT) / len(trans_errorGT))))
        print("%f" % len(trans_error))
    if args.verbose2:
        print("compared_pose_pairs %d pairs"%(len(trans_error)))
        print("absolute_translational_error.rmse %f m"%numpy.sqrt(numpy.dot(trans_error,trans_error) / len(trans_error)))
        print("absolute_translational_errorGT.rmse %f m"%numpy.sqrt(numpy.dot(trans_errorGT,trans_errorGT) / len(trans_errorGT)))

    if args.save_associations:
        file = open(args.save_associations,"w")
        file.write("\n".join(["%f %f %f %f %f %f %f %f"%(a,x1,y1,z1,b,x2,y2,z2) for (a,b),(x1,y1,z1),(x2,y2,z2) in zip(matches,first_xyz.transpose().A,second_xyz_aligned.transpose().A)]))
        file.close()
        
    if args.save:
        file = open(args.save,"w")
        file.write("\n".join(["%f "%stamp+" ".join(["%f"%d for d in line]) for stamp,line in zip(second_stamps,second_xyz_notscaled_full.transpose().A)]))
        file.close()

    if args.plot:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # обязательно, чтобы работало 3D
        import numpy as np

        # Создание 3D-фигуры
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')  # 3D-проекция

        # Функция для отображения траектории в 3D
        def plot_traj(ax, stamps, xyz, style, color, label):
            ax.plot(xyz[0], xyz[1], xyz[2], style, color=color, label=label)

        # Построение траекторий 
        # plot_traj(ax, first_stamps, first_xyz_full.A, '-', "black", "ground truth")
        # plot_traj(ax, first_stamps, first_xyz_full.transpose().A, '-', "black", "ground truth")
        # plot_traj(ax, second_stamps, second_xyz_full_aligned.A, '-', "blue", "estimated")
        plot_traj(ax, first_stamps, first_xyz_full, '-', "black", "ground truth")
        plot_traj(ax, second_stamps, second_xyz_full_aligned, '-', "blue", "estimated")

        # plot_traj(ax, second_stamps, second_xyz_full_aligned.transpose().A, '-', "blue", "estimated")
        # print(first_xyz_full)

        # Разности между точками
        # label = "difference"
        # for (a, b), (x1, y1, z1), (x2, y2, z2) in zip(matches, first_xyz.transpose().A, second_xyz_aligned.transpose().A):
        #     ax.plot([x1, x2], [y1, y2], [z1, z2], '-', color="red", label=label)
        #     label = ""  # чтобы не дублировать в легенде

        # Подписи и оформление
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        ax.legend()

        # Сохранение (если args.plot задан) и отображение
        plt.savefig(args.plot, format="pdf")  # Раскомментируй, если нужно сохранить
        plt.show()



        