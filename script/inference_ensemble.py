# inference_ensemble.py

import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import preprocessing
from my_config import config
from my_dataset import load_dataframe, StoneDataset
from my_model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Ensemble Inference Mode] Using device: {device}")

# 데이터프레임 로딩
train_df, _, test_df, _ = load_dataframe(
    config.train_dir, config.test_dir, config.test_csv_path,
    test_size=config.test_size, shuffle=config.data_shuffle
)

# 경로 보정
test_image_root = os.path.join("..", "data", "test")
test_df['img_path'] = test_df['img_path'].apply(lambda x: os.path.join(test_image_root, os.path.basename(x)))

# DataLoader
test_dataset = StoneDataset(test_df, image_size=config.image_size, transform=config.test_augmentor, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

# 라벨 인코더
le = preprocessing.LabelEncoder()
le.fit(train_df['label'])

model1 = 'resnet101'
model2 = 'xception65'
model1_path = 'path'
model2_path = 'path'

# 모델 1: resnet101
model1 = get_model(model1, num_classes=config.num_classes, pretrained=False)
model1.load_state_dict(torch.load(model1_path, map_location=device))
model1 = model1.to(device)
model1.eval()

# 모델 2: xception65
model2 = get_model(model2, num_classes=config.num_classes, pretrained=False)
model2.load_state_dict(torch.load(model2_path, map_location=device))
model2 = model2.to(device)
model2.eval()

# 🔍 추론 및 앙상블 (soft voting)
all_preds = []

with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)

        out1 = model1(images)
        out2 = model2(images)

        prob1 = F.softmax(out1, dim=1)
        prob2 = F.softmax(out2, dim=1)

        ensemble_prob = (prob1 + prob2) / 2
        preds = ensemble_prob.argmax(dim=1)

        # 클래스 인덱스를 라벨명으로 변환
        preds = le.inverse_transform(preds.cpu().numpy())
        all_preds.extend(preds)

# 📄 저장
submit = pd.read_csv('../data/sample_submission.csv')
submit['rock_type'] = all_preds
submit.to_csv('../output/ensemble_submit.csv', index=False)
print("✅ ensemble inference finished → ensemble_submit.csv")
