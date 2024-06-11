import os
import sys
# sys.path.append('/home/andrii/mmpie1/mmpie1')
import albumentations as A
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mmpie1.data.transforms import Augmentations
from mmpie1.data.preprocessor import Preprocessor
from mmpie1.data.COCODetectionDataset import CocoDetection
from mmpie1.data.MMPieDM import MMPieDataModule
import torchvision
import time
import hydra

@hydra.main(config_path='/home/andrii/mmpie1/mmpie1/configs/dataset', config_name='codetr_conf.yaml')
def main(cfg):
    
    from torchvision.transforms import ToPILImage
    topil = ToPILImage()


    dm = MMPieDataModule(cfg)

    save_path = '/home/andrii/mmpie1/debug_output/dataloader/tiled_data_val'
    os.makedirs(save_path, exist_ok=True)
    
    idx2class = {v['id']: v['name'] for v in dm.train_dataset.coco.cats.values()}

    


    for idx, e in enumerate(dm.train_dataloader()): 
        for iidx in range(e['pixel_values'].shape[0]):

            p = dm.preprocessing.renormalize(e['pixel_values'][iidx])
            p = np.array(topil(p))
            boxes = e['boxes'][iidx].numpy()
            labels = e['labels'][iidx].numpy()
            for box, label in zip(boxes, labels):
                cv2.rectangle(p, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3]),), (255, 0, 0), 2)
                cv2.putText(p, idx2class[label], (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            
            cv2.imwrite(os.path.join(save_path, f'{idx}_{iidx}.jpg'), cv2.cvtColor(p, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    main()