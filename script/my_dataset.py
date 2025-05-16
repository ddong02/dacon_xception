# dataset.py
import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

def load_dataframe(train_dir, test_dir, test_csv, test_size=0.25, shuffle=True):
    # 학습 이미지 경로와 라벨 추출
    train_image_paths = []
    train_labels = []

    for root, _, files in os.walk(train_dir):
        for file in files:
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path))
            train_image_paths.append(path)
            train_labels.append(label)

    train_df = pd.DataFrame({
        'img_path': train_image_paths,
        'label': train_labels
    })

    # 라벨 인코딩 (One-hot이 아니라 integer encoding)
    class_names = sorted(train_df['label'].unique())
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}
    train_df['label_idx'] = train_df['label'].map(class_to_idx)

    # 테스트 데이터
    test_df = pd.read_csv(test_csv)
    test_df['img_path'] = test_df['ID'].apply(lambda x: os.path.join(test_dir, f"{x}.jpg"))

    # train/val split
    train_df, val_df = train_test_split(train_df, test_size=test_size, stratify=train_df['label_idx'], shuffle=shuffle)

    return train_df, val_df, test_df, class_to_idx


class StoneDataset(Dataset):
    def __init__(self, df, image_size=(224, 224), transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['img_path']
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']

        if self.is_test:
            return image
        else:
            label = row['label_idx']
            return image, label
