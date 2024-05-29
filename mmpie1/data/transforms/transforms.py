import albumentations as A
import torchvision.transforms.v2 as T
import numpy as np
import cv2 

'''
Geometrical Transformations
    Affine
    BBoxSafeRandomCrop
    CenterCrop
    Crop
    CropAndPad
    CropNonEmptyMaskIfExists
    ElasticTransform
    Flip
    GridDistortion
    HorizontalFlip
    Perspective
    PiecewiseAffine
    RandomCrop
    RandomCropFromBorders
    RandomResizedCrop
    RandomRotate90
    RandomScale
    RandomSizedBBoxSafeCrop
    RandomSizedCrop
    Resize
    Rotate
    SafeRotate
    ShiftScaleRotate
    SmallestMaxSize
    Transpose
    VerticalFlip
Color Transformations
    CoarseDropout
    GridDropout
    MaskDropout
    MixUp
    PixelDropout
Noise Addition
    CoarseDropout
    GridDropout
    MaskDropout
    PixelDropout
Blur and Sharpen
    Morphological
Miscellaneous Transformations
    D4
    Lambda
    LongestMaxSize
    NoOp
    PadIfNeeded
    RandomGridShuffle
    XYMasking

'''



class Augmentations:
    def __init__(self, 
           HorizontalFlipProb: float = 0.5, 
           VerticalFlipProb: float = 0.5,
           ShiftLimit: float = 0.1,
           ScaleLimit: float = 0.1,
           RotateLimit: float = 30,
           ShiftScaleRotateProb: float = 0.5,
           BrightnessContrastProb: float = 0.2,
           RGBShiftProb: float = 0.2,
           HSVProb: float = 0.2,
           CLAHEProb: float = 0.1,
           GaussNoiseProb: float = 0.2,
           BlurLimit: float = 3,
           BlurProb: float = 0.5,
           MedianBlurLimit: float = 3,
           MedianBlurProb: float = 0.5,
           MinVisiability: float = 0.5,
           LongestMaxSize: int = 1333
           ):

        self.transform = A.Compose([
            A.LongestMaxSize(max_size=LongestMaxSize),
            A.OneOf([
                A.HorizontalFlip(p=HorizontalFlipProb),
                A.VerticalFlip(p=VerticalFlipProb)],
                p=0.5),
            A.ShiftScaleRotate(shift_limit=ShiftLimit, scale_limit=ScaleLimit, rotate_limit=RotateLimit, p=ShiftScaleRotateProb, 
                                border_mode=cv2.BORDER_CONSTANT, value=0
                                ),

            
            A.RandomBrightnessContrast(p=BrightnessContrastProb),
            A.RGBShift(p=RGBShiftProb),
            A.OneOf([
                A.HueSaturationValue(p=HSVProb),
                A.CLAHE(p=CLAHEProb)],
                p=0.5), 

            
            A.GaussNoise(p=GaussNoiseProb),
            A.OneOf([
                A.Blur(blur_limit=BlurLimit, p=BlurProb),
                A.MedianBlur(blur_limit=MedianBlurLimit, p=MedianBlurProb),
            ], p=0.1),
        ], bbox_params=A.BboxParams(format='coco', min_visibility=MinVisiability))

    def __call__(self, image, bboxes):
        '''

        image: np.ndarray
        bboxes: list of list

        '''
        transformed = self.transform(image=image, bboxes=bboxes)
        image = transformed['image']
        bboxes = np.array(transformed['bboxes'])
        return image, bboxes


if __name__ == '__main__':
    import cv2
    image = cv2.imread('/home/andrii/mmpie1/mmpie1/notebooks/T000002962622-BOTTOM.jpg')
    bboxes = [[0, 0, 100, 100, 0], [100, 100, 200, 200, 0]]
    aug = Augmentations()
    image, bboxes = aug(image, bboxes)
    print(image.shape, bboxes)
    