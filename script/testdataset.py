import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from my_dataset import StoneDataset  # StoneDataset 정의한 파일명
from my_config import config         # config.test_augmentor 포함
import os

# ✅ 1. test.csv 로드 및 이미지 경로 재구성
test_csv_path = '../data/test.csv'
test_image_dir = '../data/test'

test_df = pd.read_csv(test_csv_path)
test_df['img_path'] = test_df['ID'].apply(lambda x: os.path.join(test_image_dir, f"{x}.jpg"))

# ✅ 2. 상위 5개 이미지만 선택
n = 5  # 시각화할 개수
subset_df = test_df.iloc[:n].copy()

# ✅ 3. StoneDataset 생성
stone_ds = StoneDataset(subset_df, transform=config.test_augmentor, is_test=True)

# ✅ 4. 역정규화 함수
def denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

# ✅ 5. 시각화
plt.figure(figsize=(15, 4))
for i in range(n):
    img_tensor = stone_ds[i]
    img_np = denormalize(img_tensor.clone()).permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)

    plt.subplot(1, n, i + 1)
    plt.imshow(img_np)
    plt.title(f"StoneDataset {i}")
    plt.axis('off')

plt.tight_layout()
plt.show()
