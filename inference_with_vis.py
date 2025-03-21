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

# from tools.visualizer import TrackVisualizer
from visualizer_new import TrackVisualizer
from detectron2.structures import Instances
from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog

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
    # parser.add_argument('--img_path', default='/content/drive/MyDrive/Tian/Seqformer_me/ytvis_32view/test/JPEGImages/')
    # parser.add_argument('--ann_path', default='/content/drive/MyDrive/Tian/Seqformer_me/ytvis_32view/annotations/instances_test_sub.json')
    parser.add_argument('--img_path', default='/content/pic')
    parser.add_argument('--ann_path', default='/content/test.json')
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

    metadata = MetadataCatalog.get(
            "ytvis_2019_val"
    )
    metadata.thing_classes = CLASSES
    # metadata.thing_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]#red green blue
    metadata.thing_colors = [[0, 255, 0]]

    instance_mode = ColorMode.IMAGE

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    with torch.no_grad():
        torch.cuda.empty_cache()
        model, criterion, postprocessors = build_model(args)
        model.to(device)
        print(model)
        state_dict = torch.load(args.model_path)['model']
        
        model.load_state_dict(state_dict)
        model.eval()
        
        # evaluate(model, criterion, postprocessors, data_loader_val, base_ds,  device,args)

        folder = args.img_path
        videos = json.load(open(args.ann_path,'rb'))['videos']
        vis_num = len(videos)
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

            if args.save_frames:
                if not os.path.exists(args.output + '/' + file_names[0].split('/')[0]):
                    os.makedirs(args.output + '/' + file_names[0].split('/')[0])

            img_set=[]
            vid_frames = []
            vid_frame_paths = []
            for k in range(vid_len):
                im = Image.open(os.path.join(folder,file_names[k])).convert('RGB')
                im_v = read_image(os.path.join(folder,file_names[k]), format="BGR")
                vid_frames.append(im_v)
                vid_frame_paths.append(file_names[k].split('/')[-1])
                w, h = im.size
                sizes = torch.as_tensor([int(h), int(w)])
                img_set.append(transform(im).unsqueeze(0).cuda())
            img = torch.cat(img_set,0)

            model.detr.num_frames=vid_len  
            torch.cuda.empty_cache()
            
            outputs = model.inference(img,img.shape[-1],img.shape[-2])          
            logits = outputs['pred_logits'][0]
            output_mask = outputs['pred_masks_refine'][0]            
            output_box = outputs['pred_boxes'][0]

            H = output_mask.shape[-2]
            W = output_mask.shape[-1]

            scores = logits.sigmoid().cpu().detach().numpy()
            boxes = output_box.permute(1, 0, 2).sigmoid().cpu().detach().numpy()

            hit_dict={}

            topkv, indices10 = torch.topk(logits.sigmoid().cpu().detach().flatten(0),k=10)
            indices10 = indices10.tolist()
            for idx in indices10:
                queryid = idx//2
                if queryid in hit_dict.keys():
                    hit_dict[queryid].append(idx%2)
                else:
                    hit_dict[queryid]= [idx%2]
            pred_scores = []
            pred_labels = []
            pred_masks_list = []
            pred_boxes = []

            for inst_id in hit_dict.keys():
                masks = output_mask[inst_id]
                boxes = output_box.permute(1, 0, 2)[inst_id]
                pred_masks =F.interpolate(masks[:,None,:,:], (im.size[1],im.size[0]),mode="bilinear")
                pred_masks_c = pred_masks.sigmoid().cpu().detach() > 0.5 # 0.25, 0.5, 0.3
                pred_masks = pred_masks.sigmoid().cpu().detach().numpy() > 0.5  #shape [100, 36, 720, 1280] 记在result里的

                if pred_masks.max()==0:
                    # print('skip')
                    continue
                for class_id in hit_dict[inst_id]:
                    category_id = class_id
                    score =  scores[inst_id,class_id]
                    instance = {'video_id':id_, 'video_name': file_names[0][:video_name_len], 'score': float(score), 'category_id': int(category_id)}  
                    segmentation = []
                    if score > 0.5:
                        pred_scores.append(score)
                        pred_labels.append(int(category_id-1))
                        pred_masks_list.append(pred_masks_c.squeeze(1))
                        pred_boxes.append(boxes)
           
            image_size = (im.size[1],im.size[0])
            frame_masks = list(zip(*pred_masks_list))
            frame_boxes = list(zip(*pred_boxes))
    
            total_vis_output = []
            for frame_idx in range(len(vid_frames)):
                frame = vid_frames[frame_idx][:, :, ::-1]
                visualizer = TrackVisualizer(frame, metadata, instance_mode=instance_mode)
                ins = Instances(image_size)
                if len(pred_scores) > 0:
                    ins.scores = pred_scores
                    ins.pred_classes = pred_labels
                    step1mask = torch.stack(frame_masks[frame_idx], dim=0)
                    area = np.count_nonzero(step1mask.numpy())
                    print(file_names[0].split('/')[0], area)
                    # ins.pred_masks = torch.stack(frame_masks[frame_idx], dim=0)
                    if area > 500 :
                        ins.pred_masks = torch.stack(frame_masks[frame_idx], dim=0)
                    # ins.pred_masks = torch.stack(frame_masks[frame_idx], dim=0)
                    # print(area)
                    # print(ins.pred_masks)
                # print(area)
                # print(ins)
                vis_output = visualizer.draw_instance_predictions(predictions=ins)
                total_vis_output.append(vis_output)

            visualized_output = total_vis_output

            if args.save_frames:
                for img_file, _vis_output in zip(vid_frame_paths, visualized_output):
                    out_filename = os.path.join(out_vid_path, img_file)
                    print('out filename:', out_filename)
                    _vis_output.save(out_filename)
               
            H, W = visualized_output[0].height, visualized_output[0].width

            import imageio
            images = []
            for _vis_output in visualized_output:
                frame = _vis_output.get_image()#[:, :, ::-1]
                images.append(frame)

            v_name = vid_file
            imageio.mimsave(out_vid_path + v_name + ".gif", images, fps=5)

    with open(args.save_path, 'w', encoding='utf-8') as f:
        json.dump(result,f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(' inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
