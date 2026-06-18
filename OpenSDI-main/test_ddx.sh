#!/bin/bash

python test_ddx.py \
    --model MaskCLIP \
    --model_setting_name ViTL \
    --checkpoint_path "/home/Users/25_wzk/projects/OpenSDI-main/ddx_dir/MaskCLIP_sd15_20260527_18_40_38/checkpoint-4.pth" \
    --test_image_dir "/home/datasets/storage/TmpShare/wzk/DDX/test_image/image" \
    --save_json_dir "./submission_ddx" \
    --image_size 512 \
    --test_batch_size 32 \
    --num_workers 8 \
    --edge_mask_width 7
