# config.py
import cv2
from albumentations import (
    Compose, HorizontalFlip, ShiftScaleRotate, Blur, CoarseDropout, 
    RandomRain, CLAHE, ColorJitter, RandomBrightnessContrast, OneOf,
    PadIfNeeded, Resize, Normalize
)
from albumentations.pytorch.transforms import ToTensorV2

class Config:
    # 데이터 경로
    train_dir = '../data/train'
    test_dir = '../data/test'
    test_csv_path = '../data/test.csv'

    # 데이터 설정
    test_size = 0.25
    data_shuffle = True
    image_size = (224, 224)
    batch_size = 64

    # 학습 설정
    n_epochs = 50
    learning_rate = 1e-4
    num_classes = 7
    model_name = 'xception'  # 또는 'xception' (timm 필요)

    # 파일 저장
    model_save_path = '../output/best_model.pth'

    # Augmentation 설정
    train_augmentor = Compose([
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=(0.2, 0.1), scale_limit=(0.7, 1.0), rotate_limit=90, p=0.5),
        Blur(blur_limit=8, p=0.4),
        CoarseDropout(max_holes=30, max_height=16, max_width=16, p=0.2),
        OneOf([
            ColorJitter(0.2, 0.2, 0.2, 0.2, p=1.0),
            RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0)
        ], p=0.4),
        Resize(image_size[1], image_size[0]),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    test_augmentor = Compose([
        PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)),
        Resize(image_size[1], image_size[0]),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_augmentor = Compose([
    Resize(image_size[1], image_size[0]),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
    ])

config = Config() # 다른 파일에서 쉽게 쓰려고 미리 인스턴스 생성