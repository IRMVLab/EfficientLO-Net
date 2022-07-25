'''
    Provider for duck dataset from xingyu liu
'''
import math
import os
import os.path
import json
import numpy as np
import sys
import pickle
import glob
import time
from numpy.core.defchararray import join
import scipy.misc
import matplotlib.pyplot as plt

CMAP = 'plasma'

np.set_printoptions(threshold=1e10)

class SceneflowDataset():

    def __init__(self, root='/tmp/data_odometry_velodyne/dataset', NUM_POINTS = 150000, H_input = 64, W_input = 1800, train=True):
        
        self.train = train
        self.num_points = NUM_POINTS
        self.datapath = root

        self.len_list = [0, 4541, 5642, 10303, 11104, 11375, 14136, 15237, 16338, 20409, 22000, 23201, \
            24122, 25183, 28464, 29095, 30996, 32727, 33218, 35019, 40000, 40831, 43552] 
        self.file_map = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', \
            '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']

        self.T_trans = np.array([[0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, 0, 1]])

    def __getitem__(self, index):#

        for seq_idx, seq_num in enumerate(self.len_list):
            if index < seq_num:
                cur_seq = seq_idx - 1
                cur_idx_pc2 = index - self.len_list[seq_idx-1]

                if cur_idx_pc2 == 0:
                    cur_idx_pc1 = 0
                
                else:
                    cur_idx_pc1 = cur_idx_pc2 - 1        ###############    1 frame gap  ###############   
                break        

        

        Tr_path = os.path.join( 'data_odometry_calib/dataset/sequences', str(cur_seq).zfill(2), 'calib.txt')
        Tr_data = self.read_calib_file(Tr_path)
        Tr_data = Tr_data['Tr']
        Tr = Tr_data.reshape(3,4)
        Tr = np.vstack((Tr, np.array([0, 0, 0, 1.0])))
        Tr_inv = np.linalg.inv(Tr)### 

        cur_lidar_dir = os.path.join(self.datapath, self.file_map[cur_seq])

        pc1_bin = os.path.join(cur_lidar_dir, 'velodyne/' + str(cur_idx_pc1).zfill(6) + '.bin')
        pc2_bin = os.path.join(cur_lidar_dir, 'velodyne/' + str(cur_idx_pc2).zfill(6) + '.bin')

        if cur_seq > 10:
            pose = np.load('ground_truth_pose/kitti_T_diff/' + self.file_map[2] + '_diff.npy')
        else:
            pose = np.load('ground_truth_pose/kitti_T_diff/' + self.file_map[cur_seq] + '_diff.npy')

        print(pc2_bin)

        ###################################################### read the points & pose ##########################
        

        point1 = np.fromfile(pc1_bin, dtype=np.float32).reshape(-1, 4)
        point2 = np.fromfile(pc2_bin, dtype=np.float32).reshape(-1, 4)

        n1 = point1.shape[0]
        n2 = point2.shape[0]


        pos1 = np.zeros((self.num_points, 3))
        pos2 = np.zeros((self.num_points, 3))

        pos1[:n1, :3] = point1[:, :3]
        pos2[:n2, :3] = point2[:, :3]

        if cur_seq > 10:
            T_diff = np.ones((1, 12))
        else:
            T_diff = pose[cur_idx_pc2 : cur_idx_pc2 + 1, :]##### read the transform matrix


        T_diff = T_diff.reshape(3, 4)
        filler = np.array([0.0, 0.0, 0.0, 1.0])
        filler = np.expand_dims(filler, axis = 0)   ##1*4
        T_diff = np.concatenate([T_diff, filler], axis=0)#  4*4

        T_gt = np.matmul(Tr_inv, T_diff)
        T_gt = np.matmul(T_gt, Tr)
#################################################################   Change the coordinate system  ###################################################


##################################################################################################################
        
        # R_gt = T_gt[:3, :3]
        # t_gt = T_gt[:3, 3:]

        # z_gt, y_gt, x_gt = self.mat2euler( M = R_gt)
        # q_gt = self.euler2quat(z = z_gt, y = y_gt, x = x_gt)

        return pos2, pos1, n2, n1, T_gt


    def __len__(self):
        return len(self.datapath)

    def read_calib_file(self,path):  # changed

        float_chars = set("0123456789.e+- ")
        data = {}

        with open(path, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                value = value.strip()
                data[key] = value
                if float_chars.issuperset(value):
                    # try to cast to float array
                    try:
                        data[key] = np.array(list(map(float, value.split(' '))))
                    except ValueError:
                        # casting error: data[key] already eq. value, so pass
                        pass
        return data


    def mat2euler(self, M, cy_thresh=None, seq='zyx'):

        M = np.asarray(M)
        if cy_thresh is None:
            cy_thresh = np.finfo(M.dtype).eps * 4

        r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
        # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
        cy = math.sqrt(r33*r33 + r23*r23)
        if seq=='zyx':
            if cy > cy_thresh: # cos(y) not close to zero, standard form
                z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
                y = math.atan2(r13,  cy) # atan2(sin(y), cy)
                x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
            else: # cos(y) (close to) zero, so x -> 0.0 (see above)
                # so r21 -> sin(z), r22 -> cos(z) and
                z = math.atan2(r21,  r22)
                y = math.atan2(r13,  cy) # atan2(sin(y), cy)
                x = 0.0
        elif seq=='xyz':
            if cy > cy_thresh:
                y = math.atan2(-r31, cy)
                x = math.atan2(r32, r33)
                z = math.atan2(r21, r11)
            else:
                z = 0.0
                if r31 < 0:
                    y = np.pi/2
                    x = math.atan2(r12, r13)
                else:
                    y = -np.pi/2
        else:
            raise Exception('Sequence not recognized')
        return z, y, x

    def euler2quat(self, z=0, y=0, x=0, isRadian=True):
        ''' Return quaternion corresponding to these Euler angles
        Uses the z, then y, then x convention above
        Parameters
        ----------
        z : scalar
            Rotation angle in radians around z-axis (performed first)
        y : scalar
            Rotation angle in radians around y-axis
        x : scalar
            Rotation angle in radians around x-axis (performed last)
        Returns
        -------
        quat : array shape (4,)
            Quaternion in w, x, y z (real, then vector) format
        Notes
        -----
        We can derive this formula in Sympy using:
        1. Formula giving quaternion corresponding to rotation of theta radians
            about arbitrary axis:
            http://mathworld.wolfram.com/EulerParameters.html
        2. Generated formulae from 1.) for quaternions corresponding to
            theta radians rotations about ``x, y, z`` axes
        3. Apply quaternion multiplication formula -
            http://en.wikipedia.org/wiki/Quaternions#Hamilton_product - to
            formulae from 2.) to give formula for combined rotations.
        '''
    
        if not isRadian:
            z = ((np.pi)/180.) * z
            y = ((np.pi)/180.) * y
            x = ((np.pi)/180.) * x
        z = z/2.0
        y = y/2.0
        x = x/2.0
        cz = math.cos(z)
        sz = math.sin(z)
        cy = math.cos(y)
        sy = math.sin(y)
        cx = math.cos(x)
        sx = math.sin(x)
        return np.array([
                        cx*cy*cz - sx*sy*sz,
                        cx*sy*sz + cy*cz*sx,
                        cx*cz*sy - sx*cy*sz,
                        cx*cy*sz + sx*cz*sy])



if __name__ == '__main__':

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    for i in range(11):

        gt_path = str(i).zfill(2) + '_diff.txt'
    
        gt_test = np.loadtxt(gt_path)

        out_dir = 'ground_truth_pose/kitti_T_diff'

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        save_path = os.path.join(out_dir, str(i).zfill(2) + '_diff.npy')

        np.save(save_path, gt_test)


    # print(data_test.shape)
    # print(data_test[1100:1110, :])


    # dataset = SceneflowDataset( H_input = 64, W_input = 1000, train=1)
    
    # txt_list = [ '804', '1117', '205', '429', '511']

    # T_trans = np.array([[0, 0, 1, 0],
    #                     [-1, 0, 0, 0],
    #                     [0, -1, 0, 0],
    #                     [0, 0, 0, 1]])

    # for i_dir in txt_list[0:1]:

    #     bin_dir = os.path.join(BASE_DIR, 'pre_' + i_dir)
        
    #     out_dir = os.path.join(BASE_DIR, 'out_' + i_dir)

    #     if not os.path.exists(out_dir):
    #         os.mkdir(out_dir)

    #     bin_list = os.listdir(bin_dir)
    #     bin_list.sort()

    #     # for i in range(len(bin_list)):
    #     for i in range(1):

    #         out_path = os.path.join(out_dir, bin_list[i])
    #         p_path = os.path.join(bin_dir, bin_list[i])

    #         point_cloud = np.fromfile(out_path, dtype=np.float).reshape(-1, 3)

    #         np.savetxt('testfloat.txt', point_cloud)

            # print(point_cloud.shape)

            # num_point = 90000
            # n = point_cloud.shape[0]

            # pos = np.zeros((num_point, 3))
            
            # if n > num_point:
            #     print('n > 90000!!')
            #     pos = point_cloud[:num_point, :3]
            # else:
            #     pos[:n, :3] = point_cloud[:, :3]

            
            # add = np.ones((num_point, 1))
            # pos = np.concatenate([pos, add], axis = -1)


            # pos = np.matmul(T_trans, pos.T)

            # pos = pos.T[:, :3]

            # print (pos.dtype)



            # # pos.tofile(out_path)

            # if i % 100 == 0:
            #     print(i)










    # start_time = time.time()


    # for i in range(92213, 117709):
    #     pos2, pos1, n2, n1, T_gt = dataset[i]########################################


    # end_time = time.time()

    # print('read_season_time: ', end_time - start_time)






