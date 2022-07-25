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
from model_util import *


def placeholder_inputs(batch_size, NUM_POINTS):

    pointclouds = tf.placeholder(tf.float32, shape=(batch_size, NUM_POINTS * 2, 6))

    T_gt = tf.placeholder(tf.float32, shape=(batch_size, 4, 4))
    T_trans = tf.placeholder(tf.float32, shape=(batch_size, 4, 4))
    T_trans_inv = tf.placeholder(tf.float32, shape=(batch_size, 4, 4))
   
    return pointclouds, T_gt, T_trans, T_trans_inv


def get_model(point_cloud, H_input, W_input, T_gt, T_trans, T_trans_inv, is_training, bn_decay=None):

    batch_size = point_cloud.get_shape()[0].value
    num_points = point_cloud.get_shape()[1].value // 2  
    
    #####   initialize the parameters (distance  &  stride ) ######


    Down_conv_dis = [0.5, 3.0, 6.0, 12.0]
    Up_conv_dis = [3.0, 6.0, 9.0]
    Cost_volume_dis = [1.0, 2.0, 4.0]

    stride_h_list = [1, 1, 4, 2, 2, 1]
    stride_w_list = [1, 1, 8, 2, 2, 2]

    out_h_list = [math.ceil(H_input / stride_h_list[0])]
    out_w_list = [math.ceil(W_input / stride_w_list[0])]

    for i in range(1, 6):
        out_h_list.append(math.ceil(out_h_list[i - 1] / stride_h_list[i]))
        out_w_list.append(math.ceil(out_w_list[i - 1] / stride_w_list[i]))  # generate the output shape list

    ### input data  ####

    xyz_f1_input = point_cloud[:, :num_points, 0:3]
    xyz_f2_input = point_cloud[:, num_points:, 0:3]

    ### pre  process (aug + restrict)  ###

    aug_frame = np.random.choice([1, 2], size = batch_size, replace = True) # random choose aug frame 1 or 2

    xyz_f1_aug, xyz_f2_aug, q_gt, t_gt = PreProcess(xyz_f1_input, xyz_f2_input, T_gt, T_trans, T_trans_inv, aug_frame)

    xyz_f1_aug_proj, _ = ProjectPC2SphericalRing(xyz_f1_aug, None, H_input, W_input)  ## proj func
    xyz_f2_aug_proj, _ = ProjectPC2SphericalRing(xyz_f2_aug, None, H_input, W_input)

    xyz_f1_input_proj = tf.stop_gradient(xyz_f1_aug_proj)
    xyz_f2_input_proj = tf.stop_gradient(xyz_f2_aug_proj)

    points_f1_input_proj = tf.zeros((batch_size, H_input, W_input, 3))
    points_f2_input_proj = tf.zeros((batch_size, H_input, W_input, 3))

    points_f1_input_proj = tf.stop_gradient(points_f1_input_proj)
    points_f2_input_proj = tf.stop_gradient(points_f2_input_proj)

    # ####  the pre1 select bn3 xyz
    
    # pre1_selected_idx = get_selected_idx(xyz_f1_input_proj, stride_h_list[0], stride_w_list[0], out_h_list[0], out_w_list[0])  ###   b outh outw 3

    # pre1_xyz_proj_f1 = tf.gather_nd(xyz_f1_input_proj, pre1_selected_idx)  ##  b outh outw 3
    # pre1_xyz_f1 = tf.reshape(pre1_xyz_proj_f1, [batch_size, out_h_list[0] * out_w_list[0], 3])

    # pre1_xyz_proj_f2 = tf.gather_nd(xyz_f2_input_proj, pre1_selected_idx)  ##  b outh outw 3
    # pre1_xyz_f2 = tf.reshape(pre1_xyz_proj_f2, [batch_size, out_h_list[0] * out_w_list[0], 3])


    ####  the pre2 select bn3 xyz
    
    pre2_selected_idx = get_selected_idx(xyz_f1_input_proj, stride_h_list[1], stride_w_list[1], out_h_list[1], out_w_list[1])  ###   b outh outw 3
    pre2_xyz_proj_f1 = tf.gather_nd(xyz_f1_input_proj, pre2_selected_idx)  ##  b outh outw 3
    pre2_xyz_proj_f2 = tf.gather_nd(xyz_f2_input_proj, pre2_selected_idx)  ##  b outh outw 3


    ####  the l0 select bn3 xyz
    
    l0_selected_idx = get_selected_idx(pre2_xyz_proj_f1, stride_h_list[2], stride_w_list[2], out_h_list[2], out_w_list[2])  ###   b outh outw 3
    l0_xyz_proj_f1 = tf.gather_nd(pre2_xyz_proj_f1, l0_selected_idx)  ##  b outh outw 3
    l0_xyz_proj_f2 = tf.gather_nd(pre2_xyz_proj_f2, l0_selected_idx)  ##  b outh outw 3


    ####  the l1 select bn3 xyz
    
    l1_selected_idx = get_selected_idx(l0_xyz_proj_f1, stride_h_list[3], stride_w_list[3], out_h_list[3], out_w_list[3])  ###   b outh outw 3
    l1_xyz_proj_f1 = tf.gather_nd(l0_xyz_proj_f1, l1_selected_idx)  ##  b outh outw 3
    l1_xyz_proj_f2 = tf.gather_nd(l0_xyz_proj_f2, l1_selected_idx)  ##  b outh outw 3

    ####  the l2 select bn3 xyz
    
    l2_selected_idx = get_selected_idx(l1_xyz_proj_f1, stride_h_list[4], stride_w_list[4], out_h_list[4], out_w_list[4])  ###   b outh outw 3
    l2_xyz_proj_f1 = tf.gather_nd(l1_xyz_proj_f1, l2_selected_idx)  ##  b outh outw 3
    l2_xyz_proj_f2 = tf.gather_nd(l1_xyz_proj_f2, l2_selected_idx)  ##  b outh outw 3

    ####  the l3 select bn3 xyz
    
    l3_selected_idx = get_selected_idx(l2_xyz_proj_f1, stride_h_list[5], stride_w_list[5], out_h_list[5], out_w_list[5])  ###   b outh outw 3


    with tf.variable_scope('sa1') as scope:
        # Frame 1, Layer 1

        # pre1_points_f1 = pointnet_sa_module(xyz_f1_input_proj, points_f1_input_proj, pre1_xyz_f1, pre1_selected_idx, K_sample = 32, kernel_size =  [7, 15], distance = DISTANCE1, mlp = [4,4,8], mlp2 = None, flag_add=False, is_training=is_training, bn_decay=bn_decay, scope='pre1_layer')
        # pre1_points_proj_f1 = tf.reshape(pre1_points_f1, [batch_size, out_h_list[0], out_w_list[0], -1])

        # pre2_points_f1, pre2_xyz_proj_f1 = add_pointnet_sa_module(xyz_f1_input_proj, points_f1_input_proj, pre2_selected_idx, K_sample = 32, kernel_size =  [9, 15], distance = DISTANCE0_1, mlp = [4,4,8], mlp2 = None, flag_add=False, is_training=is_training, bn_decay=bn_decay, scope='pre2_layer')
        # pre2_points_proj_f1 = tf.reshape(pre2_points_f1, [batch_size, out_h_list[1], out_w_list[1], -1])

        l0_points_f1, l0_xyz_proj_f1 = down_conv(xyz_f1_input_proj, points_f1_input_proj, l0_selected_idx, \
            K_sample = 32, kernel_size =  [9, 15], distance = Down_conv_dis[0], mlp = [8,8,16], mlp2 = None, flag_add=False, is_training=is_training, bn_decay=bn_decay, scope='layer0')
        l0_points_proj_f1 = tf.reshape(l0_points_f1, [batch_size, out_h_list[2], out_w_list[2], -1])

        l1_points_f1, l1_xyz_proj_f1 = down_conv(l0_xyz_proj_f1, l0_points_proj_f1, l1_selected_idx, \
            K_sample = 32, kernel_size =  [7, 11], distance = Down_conv_dis[1], mlp = [16,16,32], mlp2 = None, flag_add=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
        l1_points_proj_f1 = tf.reshape(l1_points_f1, [batch_size, out_h_list[3], out_w_list[3], -1])

        l2_points_f1, l2_xyz_proj_f1 = down_conv(l1_xyz_proj_f1, l1_points_proj_f1, l2_selected_idx, \
            K_sample = 16, kernel_size =  [5, 9], distance = Down_conv_dis[2], mlp = [32,32,64], mlp2 = None, flag_add=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
        l2_points_proj_f1 = tf.reshape(l2_points_f1, [batch_size, out_h_list[4], out_w_list[4], -1])

        l3_points_f1, l3_xyz_proj_f1 = down_conv(l2_xyz_proj_f1, l2_points_proj_f1, l3_selected_idx, \
            K_sample = 16, kernel_size =  [5, 9], distance = Down_conv_dis[3], mlp = [64,64,128], mlp2 = None, flag_add=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')
        l3_points_proj_f1 = tf.reshape(l3_points_f1, [batch_size, out_h_list[5], out_w_list[5], -1])


        scope.reuse_variables()

        # pre1_points_f2 = pointnet_sa_module(xyz_f2_input_proj, points_f2_input_proj, pre1_xyz_f2, pre1_selected_idx, K_sample = 32, kernel_size =  [7, 15], distance = DISTANCE1, mlp = [4,4,8], mlp2 = None, flag_add=True, is_training=is_training, bn_decay=bn_decay, scope='pre1_layer')
        # pre1_points_proj_f2 = tf.reshape(pre1_points_f2, [batch_size, out_h_list[0], out_w_list[0], -1])

        # pre2_points_f2, pre2_xyz_proj_f2 = add_pointnet_sa_module(xyz_f2_input_proj, points_f2_input_proj, pre2_selected_idx, K_sample = 32, kernel_size =  [9, 15], distance = DISTANCE0_1, mlp = [4,4,8], mlp2 = None, flag_add=False, is_training=is_training, bn_decay=bn_decay, scope='pre2_layer')
        # pre2_points_proj_f2 = tf.reshape(pre2_points_f2, [batch_size, out_h_list[1], out_w_list[1], -1])

        l0_points_f2, l0_xyz_proj_f2 = down_conv(xyz_f2_input_proj, points_f2_input_proj, l0_selected_idx, \
            K_sample = 32, kernel_size =  [9, 15], distance = Down_conv_dis[0], mlp = [8,8,16], mlp2 = None, flag_add=False, is_training=is_training, bn_decay=bn_decay, scope='layer0')
        l0_points_proj_f2 = tf.reshape(l0_points_f2, [batch_size, out_h_list[2], out_w_list[2], -1])

        l1_points_f2, l1_xyz_proj_f2 = down_conv(l0_xyz_proj_f2, l0_points_proj_f2, l1_selected_idx, \
            K_sample = 32, kernel_size =  [7, 11], distance = Down_conv_dis[1], mlp = [16,16,32], mlp2 = None, flag_add=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
        l1_points_proj_f2 = tf.reshape(l1_points_f2, [batch_size, out_h_list[3], out_w_list[3], -1])

        l2_points_f2, l2_xyz_proj_f2 = down_conv(l1_xyz_proj_f2, l1_points_proj_f2, l2_selected_idx, \
            K_sample = 16, kernel_size =  [5, 9], distance = Down_conv_dis[2], mlp = [32,32,64], mlp2 = None, flag_add=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
        l2_points_proj_f2 = tf.reshape(l2_points_f2, [batch_size, out_h_list[4], out_w_list[4], -1])

        l3_points_f2, l3_xyz_proj_f2 = down_conv(l2_xyz_proj_f2, l2_points_proj_f2, l3_selected_idx, \
            K_sample = 16, kernel_size =  [5, 9], distance = Down_conv_dis[3], mlp = [64,64,128], mlp2 = None, flag_add=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')
        l3_points_proj_f2 = tf.reshape(l3_points_f2, [batch_size, out_h_list[5], out_w_list[5], -1])


    l2_xyz_f1 = tf.reshape(l2_xyz_proj_f1, [batch_size, -1, 3])
    
    l2_points_f1_new = cost_volume(l2_xyz_proj_f1, l2_xyz_proj_f2, l2_points_proj_f1, l2_points_proj_f2, \
        kernel_size1 = [3, 5], kernel_size2 = [5, 35] , nsample=4, nsample_q=32, distance = Cost_volume_dis[2], mlp1=[128,64,64], mlp2 = [128,64], \
        is_training=is_training, bn_decay=bn_decay, scope='flow_embedding_l2_origin', bn=True, pooling='max', knn=True, corr_func='concat')
    
    l2_points_new_proj_f1 =  tf.reshape(l2_points_f1_new, [batch_size, out_h_list[4], out_w_list[4], -1])  # 

    # Layer 3
    l3_points_f1_cost_volume, _ = down_conv(l2_xyz_proj_f1, l2_points_new_proj_f1, l3_selected_idx, \
        K_sample = 16, kernel_size =  [5, 9], distance = Down_conv_dis[3], mlp = [128,64,64], mlp2 = None, flag_add=False, is_training=is_training, bn_decay=bn_decay, scope='new_layer3')


#####layer3#############################################
    # Feature Propagation
    
    l3_points_predict = l3_points_f1_cost_volume
    l3_points_predict_proj = tf.reshape(l3_points_predict, [batch_size, out_h_list[5], out_w_list[5], -1])  # 

    l3_cost_volume_w = flow_predictor( l3_points_f1, None, l3_points_predict, mlp = [128,64], is_training = is_training , bn_decay = bn_decay, scope='l3_costvolume_predict_ww')
    l3_cost_volume_w_proj = tf.reshape(l3_cost_volume_w, [batch_size, out_h_list[5], out_w_list[5], -1])  # 
    
    l3_xyz_f1 = tf.reshape(l3_xyz_proj_f1, [batch_size, -1, 3])
    l3_mask_not_valid = tf.reduce_all(tf.equal(l3_xyz_f1, tf.zeros_like(l3_xyz_f1)), axis = -1)  #  B N 
    l3_mask_valid = ~l3_mask_not_valid
    
    l3_points_f1_new = softmax_valid(feature_bnc = l3_points_predict, weight_bnc = l3_cost_volume_w, mask_valid = l3_mask_valid)  # B 1 C


    l3_points_f1_new_big = tf_util.conv1d(l3_points_f1_new, 256, 1, padding='VALID', activation_fn=None, scope='l3_big')
    
    l3_points_f1_new = tf.layers.dropout(l3_points_f1_new_big, rate = 0.5, training = is_training)
    # l3_points_f1_new_t = tf.layers.dropout(l3_points_f1_new_big, rate = 0.5, training = is_training)

    l3_q_coarse = tf_util.conv1d(l3_points_f1_new, 4, 1, padding='VALID', activation_fn=None, scope='l3_q_coarse')
    l3_q_coarse = l3_q_coarse / (tf.sqrt(tf.reduce_sum(l3_q_coarse*l3_q_coarse, axis=-1, keep_dims=True)+1e-10) + 1e-10)

    l3_t_coarse = tf_util.conv1d(l3_points_f1_new, 3, 1, padding='VALID', activation_fn=None, scope='l3_t_coarse')

    l3_q = tf.squeeze(l3_q_coarse, axis = 1)
    l3_t = tf.squeeze(l3_t_coarse, axis = 1)


#####layer 2##############################################################

    l2_q_coarse = tf.reshape(l3_q, [batch_size, 1, -1])
    l2_t_coarse = tf.reshape(l3_t, [batch_size, 1, -1])
    l2_q_inv = inv_q(l2_q_coarse, batch_size)

    # # warp layer2 pose
    
    l2_xyz_f1 = tf.reshape(l2_xyz_proj_f1, [batch_size, -1, 3])
    
    l2_mask_not_valid = tf.reduce_all(tf.equal(l2_xyz_f1, tf.zeros_like(l2_xyz_f1)), axis = -1)  #  B N 
    l2_xyz1_mask = ~l2_mask_not_valid
    l2_xyz1_mask_float = tf.expand_dims(tf.to_float(l2_xyz1_mask), axis = -1)
    
    l2_xyz_bnc_q = tf.concat([tf.zeros([batch_size, out_h_list[4] * out_w_list[4], 1]), l2_xyz_f1], axis = -1)
    l2_flow_warp = mul_q_point(l2_q_coarse, l2_xyz_bnc_q, batch_size)
    l2_flow_warp = (tf.slice(mul_point_q(l2_flow_warp, l2_q_inv, batch_size), [0, 0, 1], [-1, -1, -1]) + l2_t_coarse) * l2_xyz1_mask_float

    
    ### re-project

    l2_xyz_warp_proj_f1, l2_points_warp_proj_f1 = ProjectPC2SphericalRing(l2_flow_warp, l2_points_f1, 4, 57)  # 
    l2_xyz_warp_f1 = tf.reshape(l2_xyz_warp_proj_f1, [batch_size, -1, 3])
    l2_points_warp_f1 = tf.reshape(l2_points_warp_proj_f1, [batch_size, out_h_list[4] * out_w_list[4], -1])

    l2_mask_not_valid_warp = tf.reduce_all(tf.equal(l2_xyz_warp_f1, tf.zeros_like(l2_xyz_warp_f1)), axis = -1)  #  B N 
    l2_xyz1_mask_warp = ~l2_mask_not_valid_warp

    
    # get the cost volume

    l2_cost_volume = cost_volume(l2_xyz_warp_proj_f1, l2_xyz_proj_f2, l2_points_warp_proj_f1, l2_points_proj_f2, \
        kernel_size1 = [3, 5], kernel_size2 = [5, 15] , nsample=4, nsample_q = 6, distance = Cost_volume_dis[2], mlp1=[128,64,64], mlp2 = [128,64], \
        is_training=is_training, bn_decay=bn_decay, scope='flow_embedding_l2', bn=True, pooling='max', knn=True, corr_func='concat')


    l2_cost_volume_w_up_sample = up_conv(l2_xyz_warp_proj_f1, l3_xyz_proj_f1, l2_points_warp_proj_f1, l3_cost_volume_w_proj, \
        kernel_size = [7, 15], stride_h = stride_h_list[-1], stride_w = stride_w_list[-1], nsample=8, distance = Up_conv_dis[2], mlp = [128,64], mlp2 = [128,64], scope='up_sa_layer_layer_l2w', is_training=is_training, bn_decay=bn_decay, knn=True)
    
    l2_cost_volume_up_sample = up_conv(l2_xyz_warp_proj_f1, l3_xyz_proj_f1,  l2_points_warp_proj_f1, l3_points_predict_proj, \
        kernel_size = [7, 15], stride_h = stride_h_list[-1], stride_w = stride_w_list[-1], nsample=8, distance = Up_conv_dis[2], mlp = [128,64], mlp2 = [128,64], scope='up_sa_layer_layer_l2costvolume', is_training=is_training, bn_decay=bn_decay, knn=True)

    l2_cost_volume_predict = flow_predictor( l2_points_warp_f1, l2_cost_volume_up_sample, l2_cost_volume, mlp = [128,64], is_training = is_training , bn_decay = bn_decay, scope='l2_costvolume_predict')
    l2_cost_volume_w = flow_predictor( l2_points_warp_f1, l2_cost_volume_w_up_sample, l2_cost_volume, mlp = [128,64], is_training = is_training , bn_decay = bn_decay, scope='l2_w_predict')

    l2_cost_volume_w_proj = tf.reshape(l2_cost_volume_w, [batch_size, out_h_list[4], out_w_list[4], -1])  # 
    l2_cost_volume_predict_proj = tf.reshape(l2_cost_volume_predict, [batch_size, out_h_list[4], out_w_list[4], -1])  # 

    
    #  softmax + conv1d

    l2_cost_volume_sum = softmax_valid(feature_bnc = l2_cost_volume_predict, weight_bnc = l2_cost_volume_w, mask_valid = l2_xyz1_mask_warp)  # B 1 C
    
    l2_cost_volume_sum_big = tf_util.conv1d(l2_cost_volume_sum, 256, 1, padding='VALID', activation_fn=None, scope='l2_big')
    
    l2_cost_volume_sum = tf.layers.dropout(l2_cost_volume_sum_big, rate = 0.5, training = is_training)
    # l2_cost_volume_sum_t = tf.layers.dropout(l2_cost_volume_sum_big, rate = 0.5, training = is_training)


    l2_q_det = tf_util.conv1d(l2_cost_volume_sum, 4, 1, padding='VALID', activation_fn=None, scope='l2_q_det')
    l2_q_det = l2_q_det / (tf.sqrt(tf.reduce_sum(l2_q_det*l2_q_det, axis=-1, keep_dims=True)+1e-10) + 1e-10)
    
    l2_t_det = tf_util.conv1d(l2_cost_volume_sum, 3, 1, padding='VALID', activation_fn=None, scope='l2_t_det')

    l2_t_coarse_trans = tf.concat([tf.zeros([batch_size, 1, 1]), l2_t_coarse], axis = -1)
    l2_t_coarse_trans = mul_q_point(l2_q_det, l2_t_coarse_trans, batch_size)
    l2_t_coarse_trans = tf.slice(mul_point_q(l2_t_coarse_trans, inv_q(l2_q_det, batch_size), batch_size), [0, 0, 1], [-1, -1, -1]) #### q t_coarse q_1
    
    l2_q = tf.squeeze(mul_point_q(l2_q_det, l2_q_coarse, batch_size), axis = 1)
    l2_t = tf.squeeze(l2_t_coarse_trans + l2_t_det, axis = 1)



########layer 1#####################################

    l1_q_coarse = tf.reshape(l2_q, [batch_size, 1, -1])
    l1_t_coarse = tf.reshape(l2_t, [batch_size, 1, -1])
    l1_q_inv = inv_q(l1_q_coarse, batch_size)


    # warp layer1 pose
    
    l1_xyz_f1 = tf.reshape(l1_xyz_proj_f1, [batch_size, -1, 3])

    l1_mask_not_valid = tf.reduce_all(tf.equal(l1_xyz_f1, tf.zeros_like(l1_xyz_f1)), axis = -1)  #  B N 
    l1_xyz1_mask = ~l1_mask_not_valid
    l1_xyz1_mask_float = tf.expand_dims(tf.to_float(l1_xyz1_mask), axis = -1)

    l1_xyz_bnc_q = tf.concat([tf.zeros([batch_size, out_h_list[3] * out_w_list[3], 1]), l1_xyz_f1], axis = -1)
    l1_flow_warp = mul_q_point(l1_q_coarse, l1_xyz_bnc_q, batch_size)
    l1_flow_warp = (tf.slice(mul_point_q(l1_flow_warp, l1_q_inv, batch_size), [0, 0, 1], [-1, -1, -1]) + l1_t_coarse) * l1_xyz1_mask_float

    
    # re-project
    
    l1_xyz_warp_proj_f1, l1_points_warp_proj_f1= ProjectPC2SphericalRing(l1_flow_warp, l1_points_f1, 8, 113)  # 
    l1_xyz_f1_warp = tf.reshape(l1_xyz_warp_proj_f1, [batch_size, -1, 3])
    l1_points_warp_f1 = tf.reshape(l1_points_warp_proj_f1, [batch_size, out_h_list[3] * out_w_list[3], -1])

    l1_mask_not_valid_warp = tf.reduce_all(tf.equal(l1_xyz_f1_warp, tf.zeros_like(l1_xyz_f1_warp)), axis = -1)  #  B N 
    l1_xyz1_mask_warp = ~l1_mask_not_valid_warp

    
    # get the cost volume 
    
    l1_cost_volume = cost_volume(l1_xyz_warp_proj_f1, l1_xyz_proj_f2, l1_points_warp_proj_f1, l1_points_proj_f2, \
        kernel_size1 = [3, 5], kernel_size2 = [7, 25] , nsample=4, nsample_q = 6, distance = Cost_volume_dis[1], mlp1=[128,64,64], mlp2 = [128,64], \
        is_training=is_training, bn_decay=bn_decay, scope='flow_embedding_l1', bn=True, pooling='max', knn=True, corr_func='concat')

    l1_cost_volume_w_up_sample = up_conv(l1_xyz_warp_proj_f1, l2_xyz_warp_proj_f1, l1_points_warp_proj_f1, l2_cost_volume_w_proj, \
        kernel_size = [7, 15], stride_h = stride_h_list[-2], stride_w = stride_w_list[-2], nsample=8, distance = Up_conv_dis[1], mlp = [128,64], mlp2 = [128,64], \
        scope='up_sa_layer_layer_l1w', is_training=is_training, bn_decay=bn_decay, knn=True)
    
    l1_cost_volume_up_sample = up_conv(l1_xyz_warp_proj_f1, l2_xyz_warp_proj_f1, l1_points_warp_proj_f1, l2_cost_volume_predict_proj, \
        kernel_size = [7, 15], stride_h = stride_h_list[-2], stride_w = stride_w_list[-2], nsample=8, distance = Up_conv_dis[1], mlp = [128,64], mlp2 = [128,64], \
        scope='up_sa_layer_layer_l1costvolume', is_training=is_training, bn_decay=bn_decay, knn=True)

    l1_cost_volume_predict = flow_predictor( l1_points_warp_f1, l1_cost_volume_up_sample, l1_cost_volume, mlp = [128,64], is_training = is_training , bn_decay = bn_decay, scope='l1_costvolume_predict')
    l1_cost_volume_w = flow_predictor( l1_points_warp_f1, l1_cost_volume_w_up_sample, l1_cost_volume, mlp = [128,64], is_training = is_training , bn_decay = bn_decay, scope='l1_w_predict')

    l1_cost_volume_w_proj = tf.reshape(l1_cost_volume_w, [batch_size, out_h_list[3], out_w_list[3], -1])  # 
    l1_cost_volume_predict_proj = tf.reshape(l1_cost_volume_predict, [batch_size, out_h_list[3], out_w_list[3], -1])  #


    l1_cost_volume_8 = softmax_valid(feature_bnc = l1_cost_volume_predict, weight_bnc = l1_cost_volume_w, mask_valid = l1_xyz1_mask_warp)  # B 1 C

    
    #  softmax + conv1d

    l1_cost_volume_sum_big = tf_util.conv1d(l1_cost_volume_8, 256, 1, padding='VALID', activation_fn=None, scope='l1_big')
    
    l1_cost_volume_sum = tf.layers.dropout(l1_cost_volume_sum_big, rate = 0.5, training = is_training)
    # l1_cost_volume_sum_t = tf.layers.dropout(l1_cost_volume_sum_big, rate = 0.5, training = is_training)


    l1_q_det = tf_util.conv1d(l1_cost_volume_sum, 4, 1, padding='VALID', activation_fn=None, scope='l1_q_det')
    l1_q_det = l1_q_det / (tf.sqrt(tf.reduce_sum(l1_q_det*l1_q_det, axis=-1, keep_dims=True)+1e-10) + 1e-10)
    
    l1_t_det = tf_util.conv1d(l1_cost_volume_sum, 3, 1, padding='VALID', activation_fn=None, scope='l1_t_det')

    l1_t_coarse_trans = tf.concat([tf.zeros([batch_size, 1, 1]), l1_t_coarse], axis = -1)
    l1_t_coarse_trans = mul_q_point(l1_q_det, l1_t_coarse_trans, batch_size)
    l1_t_coarse_trans = tf.slice(mul_point_q(l1_t_coarse_trans, inv_q(l1_q_det, batch_size), batch_size), [0, 0, 1], [-1, -1, -1]) #### q t_coarse q_1

    l1_q = tf.squeeze(mul_point_q(l1_q_det, l1_q_coarse, batch_size), axis = 1)
    l1_t = tf.squeeze(l1_t_coarse_trans + l1_t_det, axis = 1)


########layer 0#####################################


    l0_q_coarse = tf.reshape(l1_q, [batch_size, 1, -1])
    l0_t_coarse = tf.reshape(l1_t, [batch_size, 1, -1])
    l0_q_inv = inv_q(l0_q_coarse, batch_size)

    # warp l0
    l0_xyz_f1 = tf.reshape(l0_xyz_proj_f1, [batch_size, -1, 3])

    l0_mask_not_valid = tf.reduce_all(tf.equal(l0_xyz_f1, tf.zeros_like(l0_xyz_f1)), axis = -1)  #  B N 
    l0_xyz1_mask = ~l0_mask_not_valid
    l0_xyz1_mask_float = tf.expand_dims(tf.to_float(l0_xyz1_mask), axis = -1)
    
    l0_xyz_bnc_q = tf.concat([tf.zeros([batch_size, out_h_list[2] * out_w_list[2], 1]), l0_xyz_f1], axis = -1)
    l0_flow_warp = mul_q_point(l0_q_coarse, l0_xyz_bnc_q, batch_size)
    l0_flow_warp = (tf.slice(mul_point_q(l0_flow_warp, l0_q_inv, batch_size), [0, 0, 1], [-1, -1, -1]) + l0_t_coarse) * l0_xyz1_mask_float

    
    # re-project

    l0_xyz_warp_proj_f1, l0_points_warp_proj_f1 = ProjectPC2SphericalRing(l0_flow_warp, l0_points_f1, 16, 225)  # 
    l0_xyz_warp_f1 = tf.reshape(l0_xyz_warp_proj_f1, [batch_size, -1, 3])
    l0_points_warp_f1 = tf.reshape(l0_points_warp_proj_f1, [batch_size, out_h_list[2] * out_w_list[2], -1])

    l0_mask_not_valid_warp = tf.reduce_all(tf.equal(l0_xyz_warp_f1, tf.zeros_like(l0_xyz_warp_f1)), axis = -1)  #  B N 
    l0_xyz1_mask_warp = ~l0_mask_not_valid_warp


    # get the cost volume 

    l0_cost_volume = cost_volume(l0_xyz_warp_proj_f1, l0_xyz_proj_f2, l0_points_warp_proj_f1, l0_points_proj_f2, \
        kernel_size1 = [3, 5], kernel_size2 = [11, 41] , nsample=4, nsample_q = 6, distance = Cost_volume_dis[0], mlp1=[128,64,64], mlp2 = [128,64], \
        is_training=is_training, bn_decay=bn_decay, scope='flow_embedding_l0', bn=True, pooling='max', knn=True, corr_func='concat')

    l0_cost_volume_w_up_sample = up_conv(l0_xyz_warp_proj_f1, l1_xyz_warp_proj_f1, l0_points_warp_proj_f1, l1_cost_volume_w_proj, \
        kernel_size = [7, 15], stride_h = stride_h_list[-3], stride_w = stride_w_list[-3], nsample = 8, distance = Up_conv_dis[0], mlp = [128,64], mlp2 = [128,64], scope='up_sa_layer_layer_l0w', is_training=is_training, bn_decay=bn_decay, knn=True)
    
    l0_cost_volume_up_sample = up_conv(l0_xyz_warp_proj_f1, l1_xyz_warp_proj_f1, l0_points_warp_proj_f1, l1_cost_volume_predict_proj, \
        kernel_size = [7, 15], stride_h = stride_h_list[-3], stride_w = stride_w_list[-3], nsample = 8, distance = Up_conv_dis[0], mlp = [128,64], mlp2 = [128,64], scope='up_sa_layer_layer_l0costvolume', is_training=is_training, bn_decay=bn_decay, knn=True)

    l0_cost_volume_predict = flow_predictor( l0_points_warp_f1, l0_cost_volume_up_sample, l0_cost_volume, mlp = [128,64], is_training = is_training , bn_decay = bn_decay, scope='l0_costvolume_predict')
    l0_cost_volume_w = flow_predictor( l0_points_warp_f1, l0_cost_volume_w_up_sample, l0_cost_volume, mlp = [128,64], is_training = is_training , bn_decay = bn_decay, scope='l0_w_predict')

    
    # softmax + conv1d

    l0_cost_volume_8 = softmax_valid(feature_bnc = l0_cost_volume_predict, weight_bnc = l0_cost_volume_w, mask_valid = l0_xyz1_mask_warp)  # B 1 C

    l0_cost_volume_sum_big = tf_util.conv1d(l0_cost_volume_8, 256, 1, padding='VALID', activation_fn=None, scope='l0_big')
    
    l0_cost_volume_sum = tf.layers.dropout(l0_cost_volume_sum_big, rate = 0.5, training = is_training)
    # l0_cost_volume_sum_t = tf.layers.dropout(l0_cost_volume_sum_big, rate = 0.5, training = is_training)


    l0_q_det = tf_util.conv1d(l0_cost_volume_sum, 4, 1, padding='VALID', activation_fn=None, scope='l0_q_det')
    l0_q_det = l0_q_det / (tf.sqrt(tf.reduce_sum(l0_q_det*l0_q_det, axis=-1, keep_dims=True)+1e-10) + 1e-10)
    
    l0_t_det = tf_util.conv1d(l0_cost_volume_sum, 3, 1, padding='VALID', activation_fn=None, scope='l0_t_det')
    
    
    l0_t_coarse_trans = tf.concat([tf.zeros([batch_size, 1, 1]), l0_t_coarse], axis = -1)
    l0_t_coarse_trans = mul_q_point(l0_q_det, l0_t_coarse_trans, batch_size)
    l0_t_coarse_trans = tf.slice(mul_point_q(l0_t_coarse_trans, inv_q(l0_q_det, batch_size), batch_size), [0, 0, 1], [-1, -1, -1]) #### q t_coarse q_0

    l0_q = tf.squeeze(mul_point_q(l0_q_det, l0_q_coarse, batch_size), axis = 1)
    l0_t = tf.squeeze(l0_t_coarse_trans + l0_t_det, axis = 1)

    l0_q_norm = l0_q / (tf.sqrt(tf.reduce_sum(l0_q*l0_q, axis=-1, keep_dims=True)+1e-10) + 1e-10)
    l1_q_norm = l1_q / (tf.sqrt(tf.reduce_sum(l1_q*l1_q, axis=-1, keep_dims=True)+1e-10) + 1e-10)
    l2_q_norm = l2_q / (tf.sqrt(tf.reduce_sum(l2_q*l2_q, axis=-1, keep_dims=True)+1e-10) + 1e-10)
    l3_q_norm = l3_q / (tf.sqrt(tf.reduce_sum(l3_q*l3_q, axis=-1, keep_dims=True)+1e-10) + 1e-10)


    return  l0_q_norm, l0_t, l1_q_norm, l1_t, l2_q_norm, l2_t, l3_q_norm, l3_t, l0_xyz_f1, q_gt, t_gt



def get_loss(l0_q, l0_t, l1_q, l1_t, l2_q, l2_t, l3_q, l3_t, q_gt, t_gt, w_x, w_q):#####id

    t_gt = tf.squeeze(t_gt, axis = -1)###  8,3

    batch_size = q_gt.get_shape()[0].value


    l0_q_norm = l0_q / (tf.sqrt(tf.reduce_sum(l0_q*l0_q, axis=-1, keep_dims=True)+1e-10) + 1e-10)
    l0_loss_q = tf.reduce_mean(tf.sqrt(tf.reduce_sum((q_gt-l0_q_norm)*(q_gt-l0_q_norm), axis=-1, keep_dims=True)+1e-10)) 
    l0_loss_x = tf.reduce_mean( tf.sqrt((l0_t-t_gt) * (l0_t-t_gt)+1e-10))
    
    l0_loss = l0_loss_x * tf.exp(-w_x) + w_x + l0_loss_q * tf.exp(-w_q) + w_q
    
    tf.summary.scalar('l0 loss', l0_loss)


    l1_q_norm = l1_q / (tf.sqrt(tf.reduce_sum(l1_q*l1_q, axis=-1, keep_dims=True)+1e-10) + 1e-10)
    l1_loss_q = tf.reduce_mean(tf.sqrt(tf.reduce_sum((q_gt-l1_q_norm)*(q_gt-l1_q_norm), axis=-1, keep_dims=True)+1e-10)) 
    l1_loss_x = tf.reduce_mean(tf.sqrt((l1_t-t_gt) * (l1_t-t_gt)+1e-10))
    
    l1_loss = l1_loss_x * tf.exp(-w_x) + w_x + l1_loss_q * tf.exp(-w_q) + w_q
    
    tf.summary.scalar('l1 loss', l1_loss)


    l2_q_norm = l2_q / (tf.sqrt(tf.reduce_sum(l2_q*l2_q, axis=-1, keep_dims=True)+1e-10) + 1e-10)
    l2_loss_q = tf.reduce_mean(tf.sqrt(tf.reduce_sum((q_gt-l2_q_norm)*(q_gt-l2_q_norm), axis=-1, keep_dims=True)+1e-10))
    l2_loss_x = tf.reduce_mean(tf.sqrt((l2_t-t_gt) * (l2_t-t_gt)+1e-10))
    
    l2_loss = l2_loss_x * tf.exp(-w_x) + w_x + l2_loss_q * tf.exp(-w_q) + w_q

    tf.summary.scalar('l2 loss', l2_loss)

    l3_q_norm = l3_q / (tf.sqrt(tf.reduce_sum(l3_q*l3_q, axis=-1, keep_dims=True)+1e-10) + 1e-10)
    l3_loss_q = tf.reduce_mean(tf.sqrt(tf.reduce_sum((q_gt-l3_q_norm)*(q_gt-l3_q_norm), axis=-1, keep_dims=True)+1e-10))
    l3_loss_x = tf.reduce_mean(tf.sqrt((l3_t-t_gt) * (l3_t-t_gt)+1e-10))
    
    l3_loss = l3_loss_x * tf.exp(-w_x) + w_x + l3_loss_q * tf.exp(-w_q) + w_q

    tf.summary.scalar('l3 loss', l3_loss)

    loss_sum = 1.6*l3_loss + 0.8*l2_loss + 0.4*l1_loss + 0.2*l0_loss

    tf.add_to_collection('losses', loss_sum)
    return loss_sum



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
        


