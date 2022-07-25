python main.py \
    --mode train \
    --gpu 1 \
    --model pwclo_model \
    --data_root /tmp/data_odometry_velodyne/dataset \
    --checkpoint_path ./pretrained_model/pretrained_model.ckpt \
    --log_dir Efficient-LOnet_log_ \
    --result_dir result \
    --train_list 0 1 2 3 4 5 6 \
    --val_list 7 8 9 10 \
    --test_list 0 1 2 3 4 5 6 7 8 9 10 \
    --num_H_input 64 \
    --num_W_input 1800 \
    --max_epoch 1000 \
    --learning_rate 0.001 \
    --batch_size 8 \
    > Efficient-LOnet_log.txt 2>&1 &