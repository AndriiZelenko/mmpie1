import torchvision.transforms.v2 as T
from torchvision.transforms.v2 import functional as F
import numpy as np
import cv2



class ResizeAndPadTransform:
    def __init__(self, 
                 MinSize: int = 800, 
                 MaxSize: int = 1333):
        self.min_size = MinSize
        self.max_size = MaxSize

    def __call__(self, image, target):
        '''
        image: np.ndarray
        target: np.ndarray
        '''
        if target.size == 0:
            return image, target

        original_size = image.shape[:2] 
        h, w = original_size
        

        scale = min(self.max_size / max(h, w), self.min_size / min(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        
        
        image = cv2.resize(image, (new_w, new_h))
        
        pad_w = max(0, self.min_size - new_w)
        pad_h = max(0, self.min_size - new_h)
        top, bottom = pad_h // 2, pad_h - (pad_h // 2)
        left, right = pad_w // 2, pad_w - (pad_w // 2)
        
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        target = self.resize_boxes(target, (h, w), (new_h, new_w), top, left)
        
        return image, target

    def resize_boxes(self, target, original_size, new_size, pad_top, pad_left):
        '''
        target: np.ndarray
        original_size: tuple
        new_size: tuple
        pad_top: int
        pad_left: int
        '''
        orig_h, orig_w = original_size
        new_h, new_w = new_size
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
        
        if isinstance(target, list):
            target = np.array(target, dtype=np.float32)

        if target.ndim == 1:
            target = target.reshape(-1, 5)

        target[:, [0, 2]] = target[:, [0, 2]] * scale_x  
        target[:, [1, 3]] = target[:, [1, 3]] * scale_y  
        
        
        target[:, 0] += pad_left  
        target[:, 1] += pad_top   
        
        return target


class Preprocessor:
    def __init__(self, 
                    MinSize: int = 800,
                    MaxSize: int = 1333,
                    Mean: list = [0.485, 0.456, 0.406],
                    Std: list = [0.229, 0.224, 0.225]):
        self.min_size = MinSize
        self.max_size = MaxSize
        self.mean = Mean
        self.std = Std
        
        self.resize = ResizeAndPadTransform(MinSize=self.min_size, MaxSize=self.max_size)

        self.pipeline = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ])


    def __call__(self, image, bboxes):
        image, bboxes = self.resize(image, bboxes)
        # image = self.pipeline(image)
        return image, bboxes
    

if __name__ == "__main__":
    import time
    
    import itertools

    a = [799, 800, 801, 1333, 1334]
    

    # Generate all possible combinations
    combinations = list(itertools.product(a, a, a, a))

    image_path = "/home/andrii/mmpie1/mmpie1/notebooks/T000002962622-BOTTOM.jpg"
    image = cv2.imread(image_path)
    res = ''
    try:
        for c in combinations:
            bboxes = np.array(list(c) + [0])
        

            transform = ResizeAndPadTransform(MinSize=800, MaxSize=1333)
            start_time = time.time()
            resized_image, resized_bboxes = transform(image, bboxes)
            end_time = time.time()

            print("Processing time:", end_time - start_time)
            print("Resized image shape:", resized_image.shape)  # Should be at least (800, 800)
            res += f'{c} : {resized_image.shape}  SUCESS\n'
    except Exception as e:
        res += f'{c} :  Unsucess\n'
        print(e)
    print('result')
    print(res)









        