import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

import datasets
import util.misc as utils
from models import build_model
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
from PIL import Image
import math
import torch.nn.functional as F
import json
import pycocotools.mask as mask_util
import sys


import cv2
import time
import matplotlib.pyplot as plt

# # from tools.visualizer import TrackVisualizer
# from visualizer_new import TrackVisualizer
# from detectron2.structures import Instances
# from detectron2.data.detection_utils import read_image
# from detectron2.utils.visualizer import ColorMode
# from detectron2.data import MetadataCatalog

import torch
from engine import evaluate
from datasets import build_dataset, get_coco_api_from_dataset
from torch.utils.data import DataLoader

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--with_box_refine', default=True, action='store_true')

    # Model parameters
    parser.add_argument('--model_path', type=str, default=None,
                        help="Path to the model weights.")
    # * Backbone    
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--rel_coord', default=True, action='store_true')
    
    # Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--mask_out_stride', default=4, type=int)

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=2, type=float)
    parser.add_argument('--dice_loss_coef', default=5, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--img_path', default='/content/drive/MyDrive/Tian/SequenceFormer/ytvis_32view/test/JPEGImages')
    parser.add_argument('--ann_path', default='/content/drive/MyDrive/Tian/SequenceFormer/ytvis_32view/annotations/instances_test_sub.json')
    # parser.add_argument('--ytvis_path', default='/content/drive/MyDrive/Tian/SequenceFormer/ytvis_32view', type=str)
    parser.add_argument('--save_path', default='results.json')
    parser.add_argument('--output', default='/content')
    parser.add_argument('--dataset_file', default='YoutubeVIS')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--output_dir', default='output_ytvos',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=2, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    #parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval', action='store_false')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--num_frames', default=1, type=int, help='number of frames')
    parser.add_argument(
        "--save-frames",
        default="/content",
        help="Save frame level image outputs.",
    )
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # evaluation options
    parser.add_argument('--dataset_type', default='original')
    parser.add_argument('--eval_types', default='coco')
    parser.add_argument('--visualize', default='')
    return parser

CLASSES=['cell']

transform = T.Compose([
    T.Resize((416, 416)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def main(args):
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # metadata = MetadataCatalog.get(
    #         "ytvis_2019_val"
    # )
    # metadata.thing_classes = CLASSES
    # metadata.thing_colors = [[0, 255, 0]]
    
    # cpu_device = torch.device("cpu")
    # instance_mode = ColorMode.IMAGE

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # dataset_val = build_dataset(image_set='val', args=args)
    # sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    # data_loader_val = DataLoader(dataset_val, args.batch_size, 
    #                              sampler=sampler_val,
    #                              drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
    #                              pin_memory=True)
    # # base_ds = get_coco_api_from_dataset(dataset_val)
    # base_ds = 'coco'
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with torch.no_grad():
        torch.cuda.empty_cache()
        model, criterion, postprocessors = build_model(args)
        model.to(device)
        print(model)
        state_dict = torch.load(args.model_path)['model']
        model.load_state_dict(state_dict)
        model.eval()

        folder = args.img_path
        videos = json.load(open(args.ann_path,'rb'))['videos']
        vis_num = len(videos)
        # print(vis_num)
        result = [] 
        for i in range(vis_num):
            print("Process video: ",i)
            id_ = videos[i]['id']
            vid_len = videos[i]['length']
            file_names = videos[i]['file_names']
            video_name_len = 0 
            pred_masks = None
            pred_logits = None
            vid_file = file_names[0].split('/')[0]
            out_vid_path = os.path.join(args.output, vid_file)

            img_set=[]
            vid_frames = []
            vid_frame_paths = []
            for k in range(vid_len):
                im = Image.open(os.path.join(folder,file_names[k])).convert('RGB')
                # im_v = read_image(os.path.join(folder,file_names[k]), format="BGR")
                vid_frames.append(im)
                vid_frame_paths.append(file_names[k].split('/')[-1])

                w, h = im.size
                sizes = torch.as_tensor([int(h), int(w)])

                img_set.append(transform(im).unsqueeze(0).cuda())

            img = torch.cat(img_set,0)
            model.detr.num_frames=vid_len  
            torch.cuda.empty_cache()
            
            start_time = time.time()
            outputs = model.inference(img,img.shape[-1],img.shape[-2]) 
            end_time = time.time()
            inference_time = end_time - start_time#处理一个视频的
            fps = vis_num/inference_time
            print("inference time:",inference_time, "fps:", fps)
            # print("fps:", fps)
            



if __name__ == '__main__':
    parser = argparse.ArgumentParser(' inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
