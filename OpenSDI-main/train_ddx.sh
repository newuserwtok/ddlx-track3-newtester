#!/bin/bash

base_dir="./ddx_dir"
mkdir -p ${base_dir}

python ./train_ddx.py \
    --exp_name MaskCLIP_sd15 \
    --model_setting_name 'ViTL' \
    --model MaskCLIP \
    --world_size 1 \
    --batch_size 8 \
    --data_path "/home/datasets/storage/TmpShare/wzk/DDX/track1" \
    --epochs 5 \
    --lr 1e-4 \
    --image_size 512 \
    --if_resizing \
    --min_lr 0 \
    --weight_decay 0.05 \
    --edge_mask_width 7 \
    --if_predict_label \
    --if_not_amp \
    --test_data_path "/home/datasets/storage/TmpShare/wzk/DDX/track1" \
    --warmup_epochs 0 \
    --output_dir "./ddx_dir" \
    --log_dir "./ddx_dir" \
    --accum_iter 1 \
    --seed 42 \
    --test_period 1 \
    --num_workers 4 \
    --sample_ratio 0.01



