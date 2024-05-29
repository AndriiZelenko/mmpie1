import os
import sys
# sys.path.append('/home/andrii/mmpie1/mmpie1')
import albumentations as A
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mmpie1.data.transforms import Augmentations
from mmpie1.data.preprocessor import Preprocessor
from mmpie1.data.COCODetection import CocoDetection
import torchvision
import time
import hydra


@hydra.main(config_path='/home/andrii/mmpie1/mmpie1/configs/dataset', config_name='codetr_conf.yaml')
def main(cfg):
    print(cfg)

    augs = Augmentations(**cfg.Augmentations)
    prep = Preprocessor(**cfg.Preprocessing)

    train_dataset = CocoDetection(cfg.TrainPath, augmentations = augs, preprocessing = None, train = True, profiling_req = True, return_original=True)
    val_dataset = CocoDetection(cfg.ValPath, augmentations = augs, preprocessing = prep, train = False,  profiling_req = True, return_original=True)
    

    at_cum = 0
    aug_time_cum = 0
    get_item_time_cum = 0
    pp_cum = 0
    total = 0
    save_path = '/home/andrii/mmpie1/debug_output/dataset_check_v0/global_phone_val'
    idx2class = {0: 'HS', 1: 'LS', 2: 'D', 3:'P' , 4 : 'C', 5 : 'P'}
    min_is = float('inf')
    max_is = 0
    for idx, (i, l, p, oi, ot), in enumerate(val_dataset):
        
      
        for l1 in l:
            i = cv2.rectangle(i, (int(l1[0]), int(l1[1])), (int(l1[0] + l1[2]), int(l1[1] + l1[3])), (0,255,0), 2)
            i = cv2.putText(i, str(idx2class[l1[4]]), (int(l1[0]), int(l1[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        for o1 in ot:
            oi = cv2.rectangle(oi, (int(o1[0]), int(o1[1])), (int(o1[0] + o1[2]), int(o1[1] + o1[3])), (0,255,0), 2)
            oi = cv2.putText(oi, str(idx2class[o1[4]]), (int(o1[0]), int(o1[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

        os.makedirs(os.path.join(save_path, str(idx)), exist_ok = True)
        cv2.imwrite(os.path.join(save_path, str(idx), 'augmented.jpg'), i)
        cv2.imwrite(os.path.join(save_path, str(idx), 'original.jpg'), oi)
        if min(i.shape[0], i.shape[1]) < min_is:
            min_is = min(i.shape[0], i.shape[1])
        if max(i.shape[0], i.shape[1]) > max_is:
            max_is = max(i.shape[0], i.shape[1])

        print(f'image shape: {i.shape}')
        print('all time: ', p['whole'])
        print('---getItem time: ', p['getitem'])
        print('---augmentations time: ', p['augmentations'])
        print('---postprocessing time: ', p['preprocessing'])

        at_cum += p['whole']
        aug_time_cum += p['augmentations']
        get_item_time_cum += p['getitem']
        pp_cum += p['preprocessing']
        total +=1

    print('total: ', total)
    print('all time average: ', at_cum / total)
    print('augmentations time average: ', aug_time_cum / total)
    print('getItem time average: ', get_item_time_cum / total)
    print('postprocessing time average: ', pp_cum / total)
    print('min size: ', min_is)
    print('max size: ', max_is)
    

if __name__ == '__main__':
    main()