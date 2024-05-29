import os
import time
import numpy as np
import torchvision



class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, augmentations=None, preprocessing=None, train=True, profiling_req=False, return_original=False):
        ann_file = os.path.join(img_folder, "train.json" if train else "val.json")
        super().__init__(img_folder, ann_file)
        self.augmentations = augmentations
        self.preprocessing = preprocessing
        self.profiling_req = profiling_req
        self.return_original = return_original

    def __getitem__(self, idx):
        start_time = time.time()

        img, target = super().__getitem__(idx)
        boxes = [t['bbox'] + [t['category_id']] for t in target]

        img = np.array(img)
        boxes = np.array(boxes)
        
        # For logging might be useful
        original_image = img.copy()
        original_boxes = boxes.copy()

        augmentation_start_time = time.time()
        if self.augmentations:
            img, boxes = self.augmentations(image=img, bboxes=boxes)
        augmentation_end_time = time.time()

        if self.preprocessing:
            img, boxes = self.preprocessing(image=img, bboxes=boxes)
        end_time = time.time()

        if self.profiling_req:
            profiling = {
                'getitem': augmentation_start_time - start_time,
                'augmentations': augmentation_end_time - augmentation_start_time,
                'preprocessing': end_time - augmentation_end_time,
                'whole': end_time - start_time
            }
            if self.return_original:
                return img, boxes, profiling, original_image, original_boxes
            return img, boxes, profiling
        
        if self.return_original:
            return img, boxes, original_image, original_boxes
        return img, boxes
