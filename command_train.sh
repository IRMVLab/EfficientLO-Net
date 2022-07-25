python main.py \
    --gpu 0 \
    --model pwclo_model \
    --data /tmp/data_odometry_velodyne/dataset \
    --log_dir TEST_FIX_BUG_kitti_drop_ \
    --num_H_input 64 \
    --num_W_input 1800 \
    --max_epoch 1 \
    --learning_rate 0.001 \
    --batch_size 2 \
    > TEST_FIX_BUG_kitti_drop_.txt 2>&1 &