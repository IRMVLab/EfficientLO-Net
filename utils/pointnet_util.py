import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/2d_conv_select_k'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/2d_conv_random_k'))

from fused_conv_select_k import fused_conv_select_k
from fused_conv_random_k import fused_conv_random_k

import tensorflow as tf
import numpy as np
import tf_util


def warping_layers( xyz1, upsampled_flow):

    return xyz1 + upsampled_flow


def get_hw_idx( B, H, W):

    H_idx = tf.tile(tf.reshape(tf.range(H), [1, -1, 1, 1]), [B, 1, W, 1])
    W_idx = tf.tile(tf.reshape(tf.range(W), [1, 1, -1, 1]), [B, H, 1, 1])

    idx_q = tf.reshape(tf.concat([H_idx, W_idx], axis = -1), [B, -1, 2])

    return idx_q


def cost_volume(warped_xyz1_proj, xyz2_proj, points1_proj, points2_proj, kernel_size1, kernel_size2, nsample, nsample_q, distance, mlp1, mlp2, is_training, bn_decay, scope, bn=True, pooling='max', knn=True, corr_func='elementwise_product' ):
    
    
    with tf.variable_scope(scope) as sc:   

        B = warped_xyz1_proj.get_shape()[0].value
        H = warped_xyz1_proj.get_shape()[1].value
        W = warped_xyz1_proj.get_shape()[2].value

        warped_xyz1 = tf.reshape(warped_xyz1_proj, [B, H*W, -1])
        points1 = tf.reshape(points1_proj, [B, H*W, -1])

        random_HW_q = tf.random_shuffle(tf.range(kernel_size2[0] * kernel_size2[1]))

        idx_hw = get_hw_idx(B, H, W)  ###  B N 2

        qi_xyz_points_idx, valid_idx, valid_in_dis_idx, valid_mask = \
        fused_conv_select_k(warped_xyz1_proj, xyz2_proj, idx_hw,
        random_HW_q, H, W, H * W, kernel_size2[0], kernel_size2[1], nsample_q, flag_copy = 0, distance=1000, stride_h = 1, stride_w = 1)
        # B N M 3 idx

        qi_xyz_grouped = tf.gather_nd(xyz2_proj, qi_xyz_points_idx) * tf.stop_gradient(valid_mask)
        qi_points_grouped = tf.gather_nd(points2_proj, qi_xyz_points_idx) * tf.stop_gradient(valid_mask)

        pi_xyz_expanded = tf.tile(tf.expand_dims(warped_xyz1, 2), [1,1,nsample_q,1]) # batch_size, n_sampled, nsample, 3
        pi_points_expanded = tf.tile(tf.expand_dims(points1, 2), [1,1,nsample_q,1]) # batch_size, H, W, nsample, c
        
        pi_xyz_diff = qi_xyz_grouped - pi_xyz_expanded
        pi_euc_diff = tf.sqrt(tf.reduce_sum(tf.square(pi_xyz_diff), axis=[-1] , keep_dims=True) + 1e-20 )
        pi_xyz_diff_concat = tf.concat([pi_xyz_expanded, qi_xyz_grouped, pi_xyz_diff, pi_euc_diff], axis=-1)
        pi_xyz_diff_concat_aft_mask = pi_xyz_diff_concat ##### mask 
        
        pi_feat_diff = tf.concat(axis=-1, values=[pi_points_expanded, qi_points_grouped])
        pi_feat1_concat = tf.concat([pi_xyz_diff_concat, pi_feat_diff], axis=-1) # batch_size, npoint*m, nsample, [channel or 1] + 3
        pi_feat1_concat_aft_mask = pi_feat1_concat 

        pi_feat1_new_reshape = tf.reshape(pi_feat1_concat_aft_mask, [B, H*W, nsample_q, -1])############
        pi_xyz_diff_concat_reshape = tf.reshape(pi_xyz_diff_concat_aft_mask, [B, H*W, nsample_q, -1])####################

        for j, num_out_channel in enumerate(mlp1):
            pi_feat1_new_reshape = tf_util.conv2d(pi_feat1_new_reshape, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=True, is_training=is_training,
                                        scope='CV_%d'%(j), bn_decay=bn_decay)


        pi_xyz_encoding = tf_util.conv2d(pi_xyz_diff_concat_reshape, mlp1[-1], [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=True, is_training=is_training,
                                        scope='CV_xyz', bn_decay=bn_decay)

        pi_concat = tf.concat([pi_xyz_encoding, pi_feat1_new_reshape], axis = 3)

        for j, num_out_channel in enumerate(mlp2):
            pi_concat = tf_util.conv2d(pi_concat, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=True, is_training=is_training,
                                        scope='sum_CV_%d'%(j), bn_decay=bn_decay)

        valid_mask_bool = tf.equal(valid_mask, 1.0)
        WQ_mask = tf.tile(valid_mask_bool, multiples=[1, 1, 1, pi_concat.shape[-1]])   ####   B N K MLP[-1]    #####################  
        pi_concat_mask = tf.where(WQ_mask, pi_concat, tf.ones_like(pi_concat) * -1e10) 
        
        WQ = tf.nn.softmax(pi_concat_mask, dim=2)
        pi_feat1_new_reshape = WQ * pi_feat1_new_reshape
        pi_feat1_new_reshape_bnc = tf.reduce_sum(pi_feat1_new_reshape, axis=[2], keep_dims=False, name='avgpool_diff')#b, n, mlp1[-1]

        pi_feat1_new = tf.reshape(pi_feat1_new_reshape_bnc, [B, H, W, -1])  # project the selected central points to bhwc



        random_HW_p = tf.random_shuffle(tf.range(kernel_size1[0] * kernel_size1[1]))

        pc_xyz_points_idx, valid_idx, valid_in_dis_idx, valid_mask2 = \
        fused_conv_random_k(warped_xyz1_proj, warped_xyz1_proj, idx_hw, random_HW_p, H, W, H * W, kernel_size1[0], kernel_size1[1], \
        nsample, flag_copy = 0, distance=distance, stride_h = 1, stride_w = 1)

        pc_points_grouped = tf.gather_nd(pi_feat1_new, pc_xyz_points_idx) * tf.stop_gradient(valid_mask2)
        pc_xyz_grouped = tf.gather_nd(warped_xyz1_proj, pc_xyz_points_idx) * tf.stop_gradient(valid_mask2)


        pc_xyz_new = tf.tile( tf.expand_dims (warped_xyz1, axis = 2), [1,1,nsample,1] )
        pc_points_new = tf.tile( tf.expand_dims (points1, axis = 2), [1,1,nsample,1] )
 

        pc_xyz_diff = pc_xyz_grouped - pc_xyz_new####b , n ,m ,3
        pc_euc_diff = tf.sqrt(tf.reduce_sum(tf.square(pc_xyz_diff), axis=-1, keep_dims=True) + 1e-20)
        pc_xyz_diff_concat = tf.concat([pc_xyz_new, pc_xyz_grouped, pc_xyz_diff, pc_euc_diff], axis=-1)


        pc_xyz_encoding = tf_util.conv2d(pc_xyz_diff_concat, mlp1[-1], [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=True, is_training=is_training,
                                        scope='sum_xyz_encoding', bn_decay=bn_decay)


        pc_concat = tf.concat([pc_xyz_encoding, pc_points_new, pc_points_grouped], axis = -1)

        for j, num_out_channel in enumerate(mlp2):
            pc_concat = tf_util.conv2d(pc_concat, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=True, is_training=is_training,
                                        scope='sum_cost_volume_%d'%(j), bn_decay=bn_decay)
        
        valid_mask2_bool = tf.equal(valid_mask2, 1.0)

        WP_mask = tf.tile(valid_mask2_bool, multiples=[1, 1, 1, pc_concat.shape[-1]])   ####   B N K MLP[-1]    #####################  
        pc_concat_mask = tf.where(WP_mask, pc_concat, tf.ones_like(pc_concat) * -1e10) 
                                        
        WP = tf.nn.softmax(pc_concat_mask, dim=2)   #####  b, npoints, nsample, mlp[-1]

        pc_feat1_new = WP * pc_points_grouped

        pc_feat1_new = tf.reduce_sum(pc_feat1_new, axis=[2], keep_dims=False, name='sumpool_diff')#b*n*mlp2[-1]
        

    return  pc_feat1_new



def flow_predictor( points_f1, upsampled_feat, cost_volume, mlp, is_training, bn_decay, scope, bn=True ):

    with tf.variable_scope(scope) as sc:

        if upsampled_feat == None:
            points_concat = tf.concat(axis=-1, values=[ points_f1, cost_volume]) # B,ndataset1,nchannel1+nchannel2

        if cost_volume == None:
            points_concat = tf.concat(axis=-1, values=[ points_f1, upsampled_feat]) # B,ndataset1,nchannel1+nchannel2

        if (cost_volume != None) and (upsampled_feat != None) :
            points_concat = tf.concat(axis=-1, values=[ points_f1, upsampled_feat, cost_volume]) # B,ndataset1,nchannel1+nchannel2

        points_concat = tf.expand_dims(points_concat, 2)

        for i, num_out_channel in enumerate(mlp):
            points_concat = tf_util.conv2d(points_concat, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv_predictor%d'%(i), bn_decay=bn_decay)
        points_concat = tf.squeeze(points_concat,[2])
      
    return points_concat



def down_conv(xyz_proj, points_proj, selected_idx, K_sample, kernel_size, distance, mlp, mlp2, flag_add, is_training, bn_decay, scope, bn=True, pooling='max', knn=False, use_xyz=True, use_nchw=False):

    data_format = 'NCHW' if use_nchw else 'NHWC'
    
    with tf.variable_scope(scope) as sc:

        B = xyz_proj.get_shape()[0].value
        H = xyz_proj.get_shape()[1].value
        W = xyz_proj.get_shape()[2].value

        idx_n2 = tf.reshape(selected_idx, [B, -1, 3])   #   b h' w' 3   to   b n 3

        n_sampled = idx_n2.get_shape()[1].value
        
        random_HW = tf.random_shuffle(tf.range(kernel_size[0] * kernel_size[1]))

#######################   fused_conv   default  padding = 'same'

        new_xyz_points_idx, valid_idx, valid_in_dis_idx, valid_mask = \
        fused_conv_random_k(xyz_proj, xyz_proj, idx_n2[:, :, 1:], random_HW, H, W, n_sampled, kernel_size[0], kernel_size[1], \
        K_sample, flag_copy = 0, distance=distance, stride_h = 1, stride_w = 1)


        #output xyz  B H' W' M 3      output feature  B H' W' M C
        new_xyz_group = tf.gather_nd(xyz_proj, new_xyz_points_idx) * tf.stop_gradient(valid_mask)
        new_points_group = tf.gather_nd(points_proj, new_xyz_points_idx) * tf.stop_gradient(valid_mask)

        new_xyz_proj = tf.gather_nd(xyz_proj, selected_idx)   #  add the value to the queryed point 

        new_xyz = tf.reshape(new_xyz_proj, [B, -1, 3])
        new_xyz_expand = tf.tile(tf.expand_dims(new_xyz, 2), [1,1,K_sample,1]) # batch_size, n_sampled, kernal_total, 3
        
        xyz_diff = new_xyz_group - new_xyz_expand
        
        new_points_group_concat = tf.concat([xyz_diff, new_points_group], axis = -1) ###  concat xyz 


        # Point Feature Embedding 
        for i, num_out_channel in enumerate(mlp):
            new_points_group_concat = tf_util.conv2d(new_points_group_concat, num_out_channel, [1,1],
                                    padding='VALID', stride=[1,1],
                                    bn=bn, is_training=is_training,
                                    scope='conv%d'%(i), bn_decay=bn_decay,
                                    data_format=data_format)

        new_points_group_concat = new_points_group_concat * tf.stop_gradient(valid_mask)

        if use_nchw: new_points_group_concat = tf.transpose(new_points_group_concat, [0,2,3,1])

        # Pooling in Local Regions
        if pooling=='max':
            new_points_group_concat = tf.reduce_max(new_points_group_concat, axis=[2], keep_dims=True, name='maxpool')

        elif pooling=='avg':
            new_points_group_concat = tf.reduce_mean(new_points_group_concat, axis=[2], keep_dims=True, name='avgpool')


        # [Optional] Further Processing 
        if mlp2 is not None:
            if use_nchw: new_points_group_concat = tf.transpose(new_points_group_concat, [0,3,1,2])
            for i, num_out_channel in enumerate(mlp2):
                new_points_group_concat = tf_util.conv2d(new_points_group_concat, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_post_%d'%(i), bn_decay=bn_decay,
                                            data_format=data_format)



        new_points_group_concat = tf.squeeze(new_points_group_concat, [2]) # (B, n_sampled, mlp2[-1])
  
        return new_points_group_concat, new_xyz_proj



def up_conv(xyz1_proj, xyz2_proj, feat1_proj, feat2_proj, kernel_size, stride_h, stride_w, nsample, distance, mlp, mlp2, is_training, scope, bn_decay=None, bn=True, pooling='max', radius=None, knn=True):

    with tf.variable_scope(scope) as sc: 
        
        #  xyz1 dense points         #  xyz2 sparse points (queried)

        B = xyz1_proj.get_shape()[0].value
        H = xyz1_proj.get_shape()[1].value
        W = xyz1_proj.get_shape()[2].value
        
        xyz1 = tf.reshape(xyz1_proj, [B, H*W, -1])
        points1 = tf.reshape(feat1_proj, [B, H*W, -1])


        idx_hw = get_hw_idx(B, H, W)  ###  B N 2

        random_HW = tf.random_shuffle(tf.range(kernel_size[0] * kernel_size[1]))

        xyz1_up_xyz_points_idx, valid_idx, valid_in_dis_idx, valid_mask = fused_conv_random_k(xyz1_proj, xyz2_proj, idx_hw, \
            random_HW, H, W, H*W, kernel_size[0], kernel_size[1], \
            nsample, flag_copy = 0, distance = distance, stride_h = stride_h, stride_w = stride_w)


        xyz1_up_grouped = tf.gather_nd(xyz2_proj, xyz1_up_xyz_points_idx) * tf.stop_gradient(valid_mask)
        xyz1_up_points_grouped = tf.gather_nd(feat2_proj, xyz1_up_xyz_points_idx) * tf.stop_gradient(valid_mask)


        xyz1_expanded = tf.tile(tf.expand_dims(xyz1, 2), [1,1,nsample,1]) # batch_size, H1, W1, nsample, 3
        
        xyz1_diff = xyz1_up_grouped - xyz1_expanded
        xyz1_concat = tf.concat([xyz1_diff, xyz1_up_points_grouped], axis = -1)
        xyz1_concat_aft_mask = xyz1_concat

        xyz1_concat_aft_mask_reshape = tf.reshape(xyz1_concat_aft_mask, [B, H*W, nsample, -1])

        for j, num_out_channel in enumerate(mlp):
            xyz1_concat_aft_mask_reshape = tf_util.conv2d(xyz1_concat_aft_mask_reshape, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=True, is_training=is_training,
                                        scope='up_1_%d'%(j), bn_decay=bn_decay)

        xyz1_concat_aft_mask_reshape = xyz1_concat_aft_mask_reshape * tf.stop_gradient(valid_mask)

        
        xyz1_up_feat = tf.reduce_max(xyz1_concat_aft_mask_reshape, axis=[2], keep_dims=False, name='maxpool_up1') ## B H1*W1 C


    #############################    mlp2 

        xyz1_up_feat_concat_feat1 = tf.concat([xyz1_up_feat, points1], axis = -1)

        xyz1_up_feat_concat_feat1 = tf.expand_dims(xyz1_up_feat_concat_feat1, 2) # batch_size, n_sampled, 1, c

        for i, num_out_channel in enumerate(mlp2):
            xyz1_up_feat_concat_feat1 = tf_util.conv2d(xyz1_up_feat_concat_feat1, num_out_channel, [1,1],
                                       padding='VALID', stride=[1,1],
                                       bn=True, is_training=is_training,
                                       scope='up_2_%d'%(i), bn_decay=bn_decay)

        xyz1_up_feat_concat_feat1 = tf.squeeze(xyz1_up_feat_concat_feat1, [2]) # batch_size, npoint1, mlp2[-1]


        return xyz1_up_feat_concat_feat1

