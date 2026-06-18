#!/bin/bash

base_dir="./output_dir"
mkdir -p ${base_dir}

torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=1 \
./test.py \
    --model MaskCLIP \
    --model_setting_name 'ViTL' \
    --edge_mask_width 7 \
    --world_size 1 \
    --checkpoint_path "output_dir/MaskCLIP_sd15_20260515_15_39_30/checkpoint-9.pth" \
    --test_batch_size 8 \
    --image_size 512 \
    --if_resizing \
    --output_dir "./log/" \
    --log_dir "./log/"
