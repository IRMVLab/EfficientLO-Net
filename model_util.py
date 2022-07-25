"""
    FlowNet3D model with up convolution
"""

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import time
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import tf_util
from pointnet_util import *

def mul_q_point(q_a, q_b, batch_size):

    q_a = tf.reshape(q_a, [batch_size, 1, 4])
    
    q_result_0 = tf.multiply(q_a[ :, :, 0], q_b[ :, :, 0])-tf.multiply(q_a[ :, :, 1], q_b[ :, :, 1])-tf.multiply(q_a[ :, :, 2], q_b[ :, :, 2])-tf.multiply(q_a[ :, :, 3], q_b[ :, :, 3])
    q_result_0 = tf.reshape(q_result_0, [batch_size, -1, 1])
    
    q_result_1 = tf.multiply(q_a[ :, :, 0], q_b[ :, :, 1])+tf.multiply(q_a[ :, :, 1], q_b[ :, :, 0])+tf.multiply(q_a[ :, :, 2], q_b[ :, :, 3])-tf.multiply(q_a[ :, :, 3], q_b[ :, :, 2])
    q_result_1 = tf.reshape(q_result_1, [batch_size, -1, 1])

    q_result_2 = tf.multiply(q_a[ :, :, 0], q_b[ :, :, 2])-tf.multiply(q_a[ :, :, 1], q_b[ :, :, 3])+tf.multiply(q_a[ :, :, 2], q_b[ :, :, 0])+tf.multiply(q_a[ :, :, 3], q_b[ :, :, 1])
    q_result_2 = tf.reshape(q_result_2, [batch_size, -1, 1])

    
    q_result_3 = tf.multiply(q_a[ :, :, 0], q_b[ :, :, 3])+tf.multiply(q_a[ :, :, 1], q_b[ :, :, 2])-tf.multiply(q_a[ :, :, 2], q_b[ :, :, 1])+tf.multiply(q_a[ :, :, 3], q_b[ :, :, 0])
    q_result_3 = tf.reshape(q_result_3, [batch_size, -1, 1])

    q_result = tf.concat([q_result_0, q_result_1, q_result_2, q_result_3], axis = -1)

    return q_result   ##  B N 4


def mul_point_q(q_a, q_b, batch_size):

    q_b = tf.reshape(q_b, [batch_size, 1, 4])
    
    q_result_0 = tf.multiply(q_a[ :, :, 0], q_b[ :, :, 0])-tf.multiply(q_a[ :, :, 1], q_b[ :, :, 1])-tf.multiply(q_a[ :, :, 2], q_b[ :, :, 2])-tf.multiply(q_a[ :, :, 3], q_b[ :, :, 3])
    q_result_0 = tf.reshape(q_result_0, [batch_size, -1, 1])
    
    q_result_1 = tf.multiply(q_a[ :, :, 0], q_b[ :, :, 1])+tf.multiply(q_a[ :, :, 1], q_b[ :, :, 0])+tf.multiply(q_a[ :, :, 2], q_b[ :, :, 3])-tf.multiply(q_a[ :, :, 3], q_b[ :, :, 2])
    q_result_1 = tf.reshape(q_result_1, [batch_size, -1, 1])

    q_result_2 = tf.multiply(q_a[ :, :, 0], q_b[ :, :, 2])-tf.multiply(q_a[ :, :, 1], q_b[ :, :, 3])+tf.multiply(q_a[ :, :, 2], q_b[ :, :, 0])+tf.multiply(q_a[ :, :, 3], q_b[ :, :, 1])
    q_result_2 = tf.reshape(q_result_2, [batch_size, -1, 1])

    
    q_result_3 = tf.multiply(q_a[ :, :, 0], q_b[ :, :, 3])+tf.multiply(q_a[ :, :, 1], q_b[ :, :, 2])-tf.multiply(q_a[ :, :, 2], q_b[ :, :, 1])+tf.multiply(q_a[ :, :, 3], q_b[ :, :, 0])
    q_result_3 = tf.reshape(q_result_3, [batch_size, -1, 1])

    q_result = tf.concat([q_result_0, q_result_1, q_result_2, q_result_3], axis = -1)

    return q_result   ##  B N 4


def inv_q(q, batch_size):
    
    q = tf.squeeze(q, axis = 1)

    q_2 = tf.reduce_sum(q*q, axis = -1, keep_dims = True) + 1e-10
    q_  = tf.concat([tf.slice(q, [0, 0], [-1, 1]), -tf.slice(q, [0, 1], [-1, 3])], axis = -1)
    q_inv = q_/q_2

    return q_inv


def quatt2T(q, t):
    
    ''' Calculate rotation matrix corresponding to quaternion
    https://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/quaternions.py
    Parameters
    ----------
    q : 4 element array-like
    Returns
    -------
    M : (3,3) array
      Rotation matrix corresponding to input quaternion *q*
    Notes
    -----
    Rotation matrix applies to column vectors, and is applied to the
    left of coordinate vectors.  The algorithm here allows non-unit
    quaternions.
    References
    '''
    t0 = t[0]; t1 = t[1]; t2 = t[2]
    w = q[0]; x = q[1]; y = q[2]; z = q[3]
    Nq = w*w + x*x + y*y + z*z
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z

    c1 = tf.constant(1.0)
    add = tf.constant([[1.0, 0, 0, 0]])

    T = ([[ c1-(yY+zZ), xY-wZ, xZ+wY, t0],
            [ xY+wZ, c1-(xX+zZ), yZ-wX, t1],
            [ xZ-wY, yZ+wX, c1-(xX+yY), t2]])

    T = tf.concat([T, add], axis=0)

    return  T

def euler2quat(z, y, x):

    z = z/2.0
    y = y/2.0
    x = x/2.0
    cz = tf.cos(z)
    sz = tf.sin(z)
    cy = tf.cos(y)
    sy = tf.sin(y)
    cx = tf.cos(x)
    sx = tf.sin(x)
    return ([
                    cx*cy*cz - sx*sy*sz,
                    cx*sy*sz + cy*cz*sx,
                    cx*cz*sy - sx*cy*sz,
                    cx*cy*sz + sx*cz*sy])


def mat2euler(M, seq='zyx'):

    r11 = M[0, 0]; r12 = M[0, 1]; r13 = M[0, 2]
    r21 = M[1, 0]; r22 = M[1, 1]; r23 = M[1, 2]
    r31 = M[2, 0]; r32 = M[2, 1]; r33 = M[2, 2]

    cy = tf.sqrt(r33*r33 + r23*r23)

    z = tf.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
    y = tf.atan2(r13,  cy) # atan2(sin(y), cy)
    x = tf.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))

    return z, y, x


def AugQt(q_input, t_input, T_all, T_all_inv):

    batch_size = q_input.get_shape()[0].value
    
    for i in range(batch_size):
        
        cur_q_input = tf.reshape(q_input[i, :, :], [4])
        cur_t_input = tf.reshape(t_input[i, :, :], [3])

        cur_T_all = T_all[i, :, :]
        cur_T_all_inv = T_all_inv[i, :, :]

        cur_T0 = quatt2T(cur_q_input, cur_t_input)

        cur_T_out = tf.matmul(cur_T_all_inv, cur_T0)
        cur_T_out = tf.matmul(cur_T_out, cur_T_all)

        cur_R_out = cur_T_out[:3, :3]  ###  3 3
        cur_t_out = tf.reshape(cur_T_out[:3, 3:], [1, 1, 3])  ###  1 1 3

        z_euler, y_euler, x_euler = mat2euler(cur_R_out)
        cur_q_out = tf.reshape(euler2quat(z_euler, y_euler, x_euler), [1, 1, 4])  ####  1 1 4
        
        if i == 0:
            q_out = cur_q_out
            t_out = cur_t_out

        else:
            q_out = tf.concat([q_out, cur_q_out], axis = 0)    # b h w 3   
            t_out = tf.concat([t_out, cur_t_out], axis = 0)    # b h w 3   


    return q_out, t_out



def ProjectPC2SphericalRing(PC, Feature, H_input, W_input):

    batch_size = PC.get_shape()[0].value
    num_points = PC.get_shape()[1].value

    if Feature != None:
        num_channel = Feature.get_shape()[-1].value

    degree2radian =  math.pi / 180
    nLines = H_input    
    AzimuthResolution = 360.0 / W_input # degree
    VerticalViewDown = -24.8
    VerticalViewUp = 2.0

    # specifications of Velodyne-64
    AzimuthResolution = AzimuthResolution * degree2radian  # the original resolution is 0.18
    VerticalViewDown = VerticalViewDown * degree2radian
    VerticalViewUp = VerticalViewUp * degree2radian
    VerticalResolution = (VerticalViewUp - VerticalViewDown) / (nLines - 1)
    VerticalPixelsOffset = -VerticalViewDown / VerticalResolution

    # parameters for spherical ring's bounds

    ImgH = tf.constant(H_input)
    ImgW = tf.constant(W_input)
    
    PI = tf.constant(np.pi)
    AzimuthResolution = tf.constant(AzimuthResolution)
    VerticalPixelsOffset = tf.constant(VerticalPixelsOffset)
    VerticalResolution = tf.constant(VerticalResolution)


    for batch_idx in range(batch_size):

        ###  initialize current processed frame 
             
        cur_PC = PC[batch_idx, :, :3]  # N  3
        
        if Feature != None:
            cur_Feature = Feature[batch_idx, :, :]  # N  c

        x = cur_PC[:, 0] 
        y = cur_PC[:, 1] 
        z = cur_PC[:, 2] 
        r = tf.norm(cur_PC, ord=2, axis=1)  

        PC_project_shape = tf.constant([H_input, W_input, 3])  # shape H W 3
        if Feature != None:
            Feature_project_shape = tf.constant([H_input, W_input, num_channel])

        
        ####  get iCol & iRow

        iCol = ((PI - tf.atan2(y,x)) / AzimuthResolution) # alpha
        iCol = tf.to_int32(iCol)
        
        beta = tf.asin(z/r)                                                  # beta

        tmp_int = (beta / VerticalResolution + VerticalPixelsOffset)
        tmp_int = tf.to_int32(tmp_int)

        iRow = ImgH - tmp_int

        iRow = tf.clip_by_value(iRow, 0, H_input - 1)
        iCol = tf.clip_by_value(iCol, 0, W_input - 1)

        iRow = tf.reshape(iRow, [-1, 1])
        iCol = tf.reshape(iCol, [-1, 1])

        idx_scatter = tf.concat([iRow, iCol], axis = -1)  # N 2

        
        #  remove the same xyz points

        unique_hw, unique_idx = tf.unique(idx_scatter[:, 0]* W_input + idx_scatter[:, 1])  # N 
        num_segment = tf.shape(unique_hw) 
        
        min_r = tf.unsorted_segment_min(r, unique_idx, num_segment[0])
        min_r = tf.gather(min_r, unique_idx)
        
        mask_same = tf.where(tf.equal(r, min_r), tf.ones_like(r), tf.zeros_like(r))
        mask_same = tf.expand_dims(mask_same, axis = -1)
        
        cur_PC_unique = cur_PC * mask_same
        if Feature != None:
            cur_Feature_unique = cur_Feature * mask_same
        

        ####   Scatter points & feature 

        PC_project_current = tf.scatter_nd(idx_scatter, cur_PC_unique, PC_project_shape)
        if Feature != None:
            Feature_project_current = tf.scatter_nd(idx_scatter, cur_Feature_unique, Feature_project_shape)
        
        PC_project_current = tf.reshape(PC_project_current, [1, H_input, W_input, 3])
        PC_project_current = PC_project_current[:, :, :, :3]

        if batch_idx == 0:
            PC_project_final = PC_project_current
            if Feature != None:
                Feature_project_final = Feature_project_current

        else:
            PC_project_final = tf.concat([PC_project_final, PC_project_current], axis = 0)    # b h w 3   
            if Feature != None:
                Feature_project_final = tf.concat([Feature_project_final, Feature_project_current], axis = 0)       
      

    if Feature != None:
        return PC_project_final, Feature_project_final
    else:
        return PC_project_final, PC_project_final



def get_selected_idx(array: tf.Tensor, stride_h: int, stride_w: int, out_h: int, out_w: int):
    """According to given stride and output size, return the corresponding selected points

    Args:
        array (tf.Tensor): [any array with shape (B, H, W, 3)]
        stride_h (int): [stride in height]
        stride_w (int): [stride in width]
        out_h (int): [height of output array]
        out_w (int): [width of output array]
    Returns:
        [Tuple(tf.Tensor, tf.Tensor, tf.Tensor)]: [selected points with shape (B, out_h * out_w, 3) and (B, out_h, out_w, 3) with indices (B, N, 3)]
    """
    batch, height, width, C = array.shape
    
    height_indices = tf.tile(tf.reshape(tf.range(0, out_h * stride_h, stride_h), [1, -1, 1, 1]), [batch, 1, out_w, 1])  # b out_h out_w 1
    width_indices  = tf.tile(tf.reshape(tf.range(0, out_w * stride_w, stride_w), [1, 1, -1, 1]), [batch, out_h, 1, 1])  # b out_h out_w 1
    padding_indices = tf.tile(tf.reshape(tf.range(batch), [-1, 1, 1, 1]), [1, out_h, out_w, 1])  # b out_h out_w 1

    final_indices = tf.concat([padding_indices, height_indices, width_indices], axis = -1)  # b out_h out_w 3

    return final_indices   


def softmax_valid(feature_bnc, weight_bnc, mask_valid):

    batch_size = feature_bnc.get_shape()[0].value

    for b in range(batch_size):

        feature_bnc_current = feature_bnc[b, :, :]  ##   N C
        weight_bnc_current = weight_bnc[b, :, :]  ##   N C
        mask_valid_current = mask_valid[b, :]  ## N'

        feature_bnc_current_valid = tf.boolean_mask(feature_bnc_current, mask_valid_current)  ## N' C
        weight_bnc_current_valid = tf.boolean_mask(weight_bnc_current, mask_valid_current)  ###  N' C

        W_softmax = tf.nn.softmax(weight_bnc_current_valid, dim=0)
        feature_new_current = tf.reduce_sum(feature_bnc_current_valid * W_softmax, axis = 0, keep_dims = True)

        feature_new_current = tf.reshape(feature_new_current, [1, 1, -1])

        if b == 0:
            feature_new_final = feature_new_current
        else:
            feature_new_final = tf.concat([feature_new_final, feature_new_current], axis = 0)     #    B 1 C 

    
    return feature_new_final


def PreProcess(PC_f1, PC_f2, T_gt, T_trans, T_trans_inv, aug_frame):    ####    pre process procedure

    batch_size = PC_f1.get_shape()[0].value
    num_points = PC_f1.get_shape()[1].value

    add_T = tf.ones((batch_size, num_points, 1))
    PC_f1_concat = tf.concat([PC_f1, add_T], axis = -1)  ##  concat one to form  b n 4
    PC_f2_concat = tf.concat([PC_f2, add_T], axis = -1)  ##  concat one to form  b n 4

    #####   generate  the  valid  mask (remove the not valid points)

    mask_not_valid_f1 = tf.reduce_all(tf.equal(PC_f1, tf.zeros_like(PC_f1)), axis = -1)  #  B N 
    mask_valid_f1 = ~mask_not_valid_f1  
    mask_valid_f1 = tf.to_float(tf.expand_dims(mask_valid_f1, axis = -1)) 

    mask_not_valid_f2 = tf.reduce_all(tf.equal(PC_f2, tf.zeros_like(PC_f2)), axis = -1)  #  B N 
    mask_valid_f2 = ~mask_not_valid_f2  
    mask_valid_f2 = tf.to_float(tf.expand_dims(mask_valid_f2, axis = -1)) 

    for i in range(batch_size):

        cur_T_gt = T_gt[i, :, :]
        cur_T_trans = T_trans[i, :, :]
        cur_T_trans_inv = T_trans_inv[i, :, :]

        cur_mask_valid_f1 = mask_valid_f1[i, :, :]  #  N 1
        cur_mask_valid_f2 = mask_valid_f2[i, :, :]  #  N 1

        cur_PC_f1_concat = PC_f1_concat[i, :, :]
        cur_PC_f2_concat = PC_f2_concat[i, :, :]


        ##  select the 35m * 35m region ########

        r_f1 = tf.norm(cur_PC_f1_concat[:, :2], ord=2, axis=1)  
        cur_PC_f1_concat = tf.where( r_f1 > 35 , tf.zeros_like(cur_PC_f1_concat), cur_PC_f1_concat )
        r_f2 = tf.norm(cur_PC_f2_concat[:, :2], ord=2, axis=1)  
        cur_PC_f2_concat = tf.where( r_f2 > 35 , tf.zeros_like(cur_PC_f2_concat), cur_PC_f2_concat )


        ####  ramdomly choose the aug frame (1 or 2)   ###############

        trans = aug_frame[i]

        if trans == 2:

            ### only single aug

            cur_PC_f2_only_aug = tf.transpose(cur_PC_f2_concat, [1, 0])  ###  4 N 
            cur_PC_f2_only_aug = tf.matmul(cur_T_trans, cur_PC_f2_only_aug)
            cur_PC_f2_only_aug = tf.transpose(cur_PC_f2_only_aug, [1, 0])  ### N 4


            cur_PC_f1_aft_aug = cur_PC_f1_concat[:, :3]
            cur_PC_f2_aft_aug = cur_PC_f2_only_aug[:, :3]

            cur_T_gt = tf.matmul(cur_T_trans, cur_T_gt)


        elif trans == 1:

            ### only single aug

            cur_PC_f1_only_aug = tf.transpose(cur_PC_f1_concat, [1, 0])  ###  4 N 
            cur_PC_f1_only_aug = tf.matmul(cur_T_trans, cur_PC_f1_only_aug)
            cur_PC_f1_only_aug = tf.transpose(cur_PC_f1_only_aug, [1, 0])  ### N 4


            cur_PC_f1_aft_aug = cur_PC_f1_only_aug[:, :3]
            cur_PC_f2_aft_aug = cur_PC_f2_concat[:, :3]

            cur_T_gt = tf.matmul(cur_T_gt, cur_T_trans_inv)

        cur_PC_f1_aft_aug = cur_PC_f1_aft_aug * cur_mask_valid_f1
        cur_PC_f2_aft_aug = cur_PC_f2_aft_aug * cur_mask_valid_f2

        cur_R_gt = cur_T_gt[:3, :3]  ###  3 3
        cur_t_gt = tf.expand_dims(cur_T_gt[:3, 3:], axis = 0)  ###  1 3 1

        z_euler, y_euler, x_euler = mat2euler(cur_R_gt)
        cur_q_gt = tf.expand_dims(euler2quat(z_euler, y_euler, x_euler), axis = 0)  ####  1 4
        
        cur_PC_f1_aft_aug = tf.expand_dims(cur_PC_f1_aft_aug, axis = 0)
        cur_PC_f2_aft_aug = tf.expand_dims(cur_PC_f2_aft_aug, axis = 0)


        if i == 0:
            PC_f1_aft_aug = cur_PC_f1_aft_aug
            PC_f2_aft_aug = cur_PC_f2_aft_aug
            q_gt = cur_q_gt
            t_gt = cur_t_gt

        else:
            PC_f1_aft_aug = tf.concat([PC_f1_aft_aug, cur_PC_f1_aft_aug], axis = 0)    # b h w 3   
            PC_f2_aft_aug = tf.concat([PC_f2_aft_aug, cur_PC_f2_aft_aug], axis = 0)   # b n 3          
            q_gt = tf.concat([q_gt, cur_q_gt], axis = 0)          
            t_gt = tf.concat([t_gt, cur_t_gt], axis = 0)   


    return PC_f1_aft_aug, PC_f2_aft_aug, q_gt, t_gt



if __name__=='__main__':

    with tf.device('/gpu:1'):
        
        PC_project_shape = tf.constant([5, 5, 3])

        r = tf.constant([9.0, 7.0, 7.0, 9.0, 8.0])
        PC_idx = tf.ones((5, 3))

        iRow = tf.constant([[1],[1],[1],[1],[4]])
        iCol = tf.constant([[1],[1],[1],[3],[4]])

        idx_scatter = tf.concat([iRow, iCol], axis = -1)  # N 2

        unique_hw, unique_idx = tf.unique(idx_scatter[:, 0]*1800 + idx_scatter[:, 1])  # N 
        num_segment = tf.shape(unique_hw) 
        
        min_r = tf.unsorted_segment_min(r, unique_idx, num_segment[0])
        min_r = tf.gather(min_r, unique_idx)
        
        mask_same = tf.where(tf.equal(r, min_r), tf.ones_like(r), tf.zeros_like(r))
        mask_same = tf.expand_dims(mask_same, axis = -1)
        PC_idx_unique = PC_idx * mask_same

        PC_project_current = tf.scatter_nd(idx_scatter, PC_idx_unique, PC_project_shape)
    
    with tf.Session('') as sess:

        result_1 = sess.run(min_r)
        result_2 = sess.run(PC_project_current)


        print('pc_project', result_2)
        print('min_r', result_1)
        


