# -*- coding:UTF-8 -*-

import os
import yaml
import argparse
import numpy as np
import torch.utils.data as data

from tools.euler_tools import euler2quat, mat2euler
from tools.points_process import aug_matrix, limited_points, filter_points

# author:Zhiheng Feng
# contact: fzhsjtu@foxmail.com
# datetime:2021/10/21 20:04
# software: PyCharm

"""
文件说明：数据集读取

"""


class points_dataset(data.Dataset):

    def __init__(self, is_training: int = 1, num_point: int = 150000, data_dir_list: list = [0, 1, 2, 3, 4, 5, 6], config: argparse.Namespace = None):
        """

        :param train: 0训练集，1验证集，2测试集
        :param data_dir_list: 数据集序列
        :param config: 配置参数
        """

        data_dir_list.sort()
        self.num_point = num_point
        self.is_training = is_training
        self.args = config
        self.data_list = data_dir_list
        self.lidar_root = config.data_root
        self.data_len_sequence = [4540, 1100, 4660, 800, 270, 2760, 1100, 1100, 4070, 1590, 1200]

        Tr_tmp = []
        data_sum = [0]
        vel_to_cam_Tr = []

        with open('./tools/calib.yaml', "r") as f:
            con = yaml.load(f, Loader=yaml.FullLoader)
        for i in range(11):
            vel_to_cam_Tr.append(np.array(con['Tr{}'.format(i)]))
        for i in self.data_list:
            data_sum.append(data_sum[-1] + self.data_len_sequence[i] + 1)
            Tr_tmp.append(vel_to_cam_Tr[i])

        self.Tr_list = Tr_tmp
        self.data_sum = data_sum
        self.lidar_path = self.lidar_root

    def __len__(self):
        return self.data_sum[-1]

    def __getitem__(self, index):

        sequence_str_list = []

        for item in self.data_list:
            sequence_str_list.append('{:02d}'.format(item))

        if index in self.data_sum:
            index_index = self.data_sum.index(index)
            index_ = 0
            fn1 = index_
            fn2 = index_
        else:
            index_index, data_begin, data_end = self.get_index(index, self.data_sum)
            index_ = index - data_begin
            fn1 = index_ - 1
            fn2 = index_

        Tr = self.Tr_list[index_index]
        Tr_inv = np.linalg.inv(Tr)### 
        
        pose_path = 'pose/' + sequence_str_list[index_index] + '_diff.npy'
        lidar_path = os.path.join(self.lidar_path, sequence_str_list[index_index], 'velodyne')
        pose = np.load(pose_path)

        fn1_dir = os.path.join(lidar_path, '{:06d}.bin'.format(fn1))
        fn2_dir = os.path.join(lidar_path, '{:06d}.bin'.format(fn2))
        point1 = np.fromfile(fn1_dir, dtype=np.float32).reshape(-1, 4)
        point2 = np.fromfile(fn2_dir, dtype=np.float32).reshape(-1, 4)

        pos1 = np.zeros((self.num_point, 3))
        pos2 = np.zeros((self.num_point, 3))

        pos1[ :point1.shape[0], :] = point1[:, :3]
        pos2[ :point2.shape[0], :] = point2[:, :3]

        T_diff = pose[index_:index_ + 1, :]  ##### read the transformation matrix

        T_diff = T_diff.reshape(3, 4)
        filler = np.array([0.0, 0.0, 0.0, 1.0])
        filler = np.expand_dims(filler, axis=0)  ##1*4
        T_diff = np.concatenate([T_diff, filler], axis=0)  # 4*4

        T_gt = np.matmul(Tr_inv, T_diff)
        T_gt = np.matmul(T_gt, Tr)

        pos1 = pos1.astype(np.float32)
        pos2 = pos2.astype(np.float32)
        
        if self.is_training:
            T_trans = aug_matrix()
        else:
            T_trans = np.eye(4).astype(np.float32)

        T_trans_inv = np.linalg.inv(T_trans)

        return pos2, pos1, T_gt, T_trans, T_trans_inv, Tr

    def get_index(self, value, mylist):
        mylist.sort()
        for i, num in enumerate(mylist):
            if num > value:
                return i - 1, mylist[i - 1], num
