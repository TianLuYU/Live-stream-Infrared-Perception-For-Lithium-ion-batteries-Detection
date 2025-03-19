#!/usr/bin/env bash

set -x

python3 -u main.py \
    --dataset_file YoutubeVIS \
    --epochs 100 \
    --lr 2e-4 \
    --lr_drop 2 10\
    --batch_size 2 \
    --num_workers 0 \
    --coco_path ../coco \
    --ytvis_path /content/drive/MyDrive/Tian/Seqformer_me/ytvis_32view \
    --num_queries 300 \
    --num_frames 1 \
    --with_box_refine \
    --masks \
    --rel_coord \
    --backbone resnet101 \
    --pretrain_weights /content/drive/MyDrive/Tian/Seqformer_me/weights/r101_weight.pth \
    --output_dir r101_1.14_100 \


