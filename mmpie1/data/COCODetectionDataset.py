import os
import time
import json
import numpy as np
import torchvision



class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, augmentations=None, preprocessing=None, train=True, profiling_req=False, return_original=False):
        ann_file = os.path.join(img_folder, "labels.json")
        super().__init__(img_folder, ann_file)
        with open(ann_file, 'r') as f:
            self.json_file = json.load(f)
        self.idx_2_class = {v['id']: v['name'] for  v in self.json_file['categories']}
        self.augmentations = augmentations
        self.preprocessing = preprocessing
        self.profiling_req = profiling_req
        self.return_original = return_original

    def __getitem__(self, idx):
        start_time = time.time()

        img, labels = super().__getitem__(idx)
        image_size = img.size

        boxes = [t['bbox'] + [t['category_id']] for t in labels]
        mask = np.ones((img.size[1], img.size[0]), dtype=np.float32)
        img = np.array(img)
        boxes = np.array(boxes) 
        
 
        if self.return_original:
            original_image = img.copy()
            original_boxes = boxes.copy()
        else:
            original_image = None
            original_boxes = boxes.copy()


        augmentation_start_time = time.time()
        if self.augmentations:
            img, boxes = self.augmentations(image=img, bboxes=boxes)
            
        augmentation_end_time = time.time()
        if self.preprocessing:
            img, boxes,labels, mask = self.preprocessing(image=img, bboxes=boxes)
        end_time = time.time()


        
        if self.profiling_req:
            profiling = {
                'getitem': augmentation_start_time - start_time,
                'augmentations': augmentation_end_time - augmentation_start_time,
                'preprocessing': end_time - augmentation_end_time,
                'whole': end_time - start_time
            }
        else:
            profiling = {}

        data = {'pixel_values': img, 'pixel_mask': mask,
                'boxes': boxes, 'class_labels': labels,
                'image_id': idx, 'original_image': original_image, 'original_boxes': original_boxes, 'original_size':image_size, 
                'profiling': profiling}


        
        # {'size': tensor([ 800, 1066]), 
        #  'image_id': tensor([0]), 
        #  'class_labels': tensor([0]), 
        #  'boxes': tensor([[0.5955, 0.5811, 0.2202, 0.3561]]), 
        #  'area': tensor([3681.5083]), 
        #  'iscrowd': tensor([0]), 
        #  'orig_size': tensor([1536, 2048])}
        # target = {'size': [img.shape[0], img.shape[1]], 'labels': labels, 'boxes': boxes, 'pixel_mask': mask}
        return data 
    
    # TODO: implement this method
    def calculate_stats(self):
        print('TODO datasets stats calculation')
        pass
