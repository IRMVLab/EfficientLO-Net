python main.py \
    --gpu 0 \
    --model pwclo_model \
    --data /tmp/data_odometry_velodyne/dataset \
    --log_dir LOG_DIR \
    --num_H_input 64 \
    --num_W_input 1800 \
    --max_epoch 1000 \
    --learning_rate 0.001 \
    --batch_size 8 \
    > LOG_DIR.txt 2>&1 &