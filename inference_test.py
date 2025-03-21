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

from pycocotools.ytvos import YTVOS
from pycocotools.ytvoseval import YTVOSeval

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def ytvos_eval(result_file, result_types, ytvos, max_dets=(100, 300, 1000), get_boundary=False):
    
    ytvos = YTVOS(ytvos, get_boundary)
    assert isinstance(ytvos, YTVOS)

    if len(ytvos.anns) == 0:
        print("Annotations does not exist")
        return
    
    assert result_file.endswith('.json')
    ytvos_dets = ytvos.loadRes(result_file)

    vid_ids = ytvos.getVidIds()
    for res_type in result_types:
        iou_type = res_type
        ytvosEval = YTVOSeval(ytvos, ytvos_dets, iou_type)
        ytvosEval.params.vidIds = vid_ids
        if res_type == 'proposal':
            ytvosEval.params.useCats = 0
            ytvosEval.params.maxDets = list(max_dets)
        ytvosEval.evaluate()
        ytvosEval.accumulate()
        ytvosEval.summarize()



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
    parser.add_argument('--img_path', default='/content/drive/MyDrive/Tian/SequenceFormer/ytvis_32view/test/JPEGImages/')
    parser.add_argument('--ann_path', default='/content/drive/MyDrive/Tian/SequenceFormer/ytvis_32view/annotations/instances_test_sub.json')
    parser.add_argument('--save_path', default='Val_results.json')
    parser.add_argument('--dataset_file', default='YoutubeVIS')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='output_ytvos',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=21, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    #parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval', action='store_false')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--num_frames', default=1, type=int, help='number of frames')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser

# CLASSES=['person','giant_panda','lizard','parrot','skateboard','sedan','ape',
#          'dog','snake','monkey','hand','rabbit','duck','cat','cow','fish',
#          'train','horse','turtle','bear','motorbike','giraffe','leopard',
#          'fox','deer','owl','surfboard','airplane','truck','zebra','tiger',
#          'elephant','snowboard','boat','shark','mouse','frog','eagle','earless_seal',
#          'tennis_racket']

CLASSES=['cell']

transform = T.Compose([
    T.Resize(417),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])




def main(args):
    torch.cuda.empty_cache()
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model_floder = "/content/drive/MyDrive/Tian/Seqformer_me/r101_1.14_100"
    with torch.no_grad():
        model, _, postprocessors = build_model(args)
        model.to(device)
        model_paths = os.listdir(model_floder)
        # model_paths.sort()
        model_paths.reverse()
        for checkpoint in model_paths:
            print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
            print("Loading:", checkpoint)
            state_dict = torch.load(os.path.join(model_floder, checkpoint))['model']
            model.load_state_dict(state_dict)
            model.eval()
            folder = args.img_path
            videos = json.load(open(args.ann_path,'rb'))['videos']#[:5]
            vis_num = len(videos)
            result = [] 
            for i in range(vis_num):
                # print("Process video: ",i)
                id_ = videos[i]['id']
                vid_len = videos[i]['length']
                vid_height = videos[i]['height']
                vid_width = videos[i]['width']
                file_names = videos[i]['file_names']
                # video_name_len = 13 

                pred_masks = None
                pred_logits = None

                img_set=[]
                for k in range(vid_len):
                    im = Image.open(os.path.join(folder,file_names[k]))
                    w, h = im.size
                    sizes = torch.as_tensor([int(h), int(w)])
                    img_set.append(transform(im).unsqueeze(0).cuda())

                img = torch.cat(img_set,0)
          
                model.detr.num_frames=vid_len  

                outputs = model.inference(img,img.shape[-1],img.shape[-2])
                logits = outputs['pred_logits'][0]
                output_mask = outputs['pred_masks_refine'][0]
                output_boxes = outputs['pred_boxes'][0]
  
            
                H = output_mask.shape[-2]
                W = output_mask.shape[-1]


                scores = logits.sigmoid().cpu().detach().numpy()
                hit_dict={}

                topkv, indices10 = torch.topk(logits.sigmoid().cpu().detach().flatten(0),k=1)
                indices10 = indices10.tolist()
                for idx in indices10:
                    queryid = idx//2
                    if queryid in hit_dict.keys():
                        hit_dict[queryid].append(idx%2)
                    else:
                        hit_dict[queryid]= [idx%2]


                for inst_id in hit_dict.keys():
                    masks = output_mask[inst_id]
                    pred_masks =F.interpolate(masks[:,None,:,:], (im.size[1],im.size[0]),mode="bilinear")
                    # pred_masks_c = pred_masks.sigmoid().cpu().detach() > 0.9
                    # pred_masks_c2 = torch.where(pred_masks_c, 1, 0).numpy()
                    # area = np.count_nonzero(pred_masks_c2)
                    # print(area)
                    pred_masks = pred_masks.sigmoid().cpu().detach().numpy()>0.9  #shape [100, 36, 720, 1280]
                    
                    if pred_masks.max()==0:
                        # print('skip')
                        continue
                    for class_id in hit_dict[inst_id]:
                        category_id = class_id
                        score =  scores[inst_id,class_id]
                        instance = {'height': vid_height, 'width':vid_width, 'video_id':id_, 'video_name': file_names[0].split('/')[0], 'score': float(score), 'category_id': int(category_id)}  
                        # print(file_names[0].split('/')[0])
                        segmentation = []
                        for n in range(vid_len):
                            if score < 0.001:
                                segmentation.append(None)
                            else:
                                maskc = (pred_masks[n,0]).astype(np.uint8)
                                area = np.count_nonzero(maskc)
                                # print(area)
                                if area > 1500:#1500
                                    mask = (pred_masks[n,0]).astype(np.uint8)
                                    T = np.array(mask[:,:,np.newaxis], order='F')
                                    rle = mask_util.encode(T)[0]
                                    rle["counts"] = rle["counts"].decode("utf-8")
                                    segmentation.append(rle)
                                else:
                                    segmentation.append(None)
                            # segmentation.append(mask)
                        instance['segmentations'] = segmentation
                        result.append(instance)
                # frame_masks = list(zip(*pred_masks_list))        
            # json_filename = f"checkpoint.json"
            with open(args.save_path, 'w', encoding='utf-8') as f: 
                json.dump(result, f, indent=4)
            result_file = args.save_path
            ytvos_eval(result_file, ['bbox'], '/content/drive/MyDrive/Tian/SequenceFormer/ytvis_32view/annotations/instances_test_sub.json', max_dets=(100, 300, 1000), get_boundary=True)
            ytvos_eval(result_file, ['segm'], '/content/drive/MyDrive/Tian/SequenceFormer/ytvis_32view/annotations/instances_test_sub.json', max_dets=(100, 300, 1000), get_boundary=False)
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(' inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
