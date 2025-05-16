import pandas as pd
import numpy as np
import os
import glob
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
from tqdm.auto import tqdm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import timm
import warnings

warnings.filterwarnings("ignore")

# ====== PadSquare & Dataset 정의 ======

class PadSquare(A.ImageOnlyTransform):
    def __init__(self, border_mode=0, value=0, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.border_mode = border_mode
        self.value = value

    def apply(self, image, **params):
        h, w, c = image.shape
        max_dim = max(h, w)
        pad_h = max_dim - h
        pad_w = max_dim - w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.value)

    def get_transform_init_args_names(self):
        return ("border_mode", "value")

class CustomDataset(Dataset):
    def __init__(self, img_path_list, transforms=None):
        self.img_path_list = img_path_list
        self.transforms = transforms
        
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        image = cv2.imread(img_path)
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        return image
        
    def __len__(self):
        return len(self.img_path_list)

# ====== Inference 함수 ======

def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs in tqdm(test_loader, desc="Inferencing"):
            imgs = imgs.float().to(device)
            pred = model(imgs)
            preds += pred.argmax(1).detach().cpu().numpy().tolist()
    return preds

# ====== Main ======

if __name__ == '__main__':
    IMG_SIZE = 224
    BATCH_SIZE = 64
    MODEL_PATH = './output/models/model_epoch39.pth'
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # ===== LabelEncoder 학습 (train 디렉토리 기반) =====
    all_img_list = glob.glob('./data/train/*/*')
    df = pd.DataFrame()
    df['img_path'] = all_img_list
    df['rock_type'] = df['img_path'].apply(lambda x: os.path.basename(os.path.dirname(x)))

    le = preprocessing.LabelEncoder()
    le.fit(df['rock_type'])

    # ===== Test 데이터셋 및 변환 =====
    test_df = pd.read_csv('./data/test.csv')

    test_transform = A.Compose([
        PadSquare(value=(0, 0, 0)),
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    test_dataset = CustomDataset(test_df['img_path'].values, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ===== 모델 로드 및 예측 =====
    model = timm.create_model('xception', pretrained=False, num_classes=len(le.classes_))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    preds = inference(model, test_loader, DEVICE)
    decoded_preds = le.inverse_transform(preds)

    # ===== 제출 파일 저장 =====
    submit = pd.read_csv('./data/sample_submission.csv')
    submit['rock_type'] = decoded_preds
    submit.to_csv('./data/baseline_submit.csv', index=False)