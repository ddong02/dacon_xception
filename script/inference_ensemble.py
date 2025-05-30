import os
import pandas as pd
import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import preprocessing
from my_config import config
from my_dataset import load_dataframe, StoneDataset
from my_model import get_model
from tqdm import tqdm

class ConvNeXtV2WithDropout(nn.Module):
    def __init__(self, num_classes=7, dropout_p=0.3):
        super().__init__()
        self.model = timm.create_model("convnextv2_tiny", pretrained=False)
        in_features = self.model.head.fc.in_features
        self.model.head.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

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

model1_path = r"D:\dh\Dacon\Xception\resnet101_model_epoch_50.pth"
model2_path = r"D:\dh\Dacon\Xception\output\best_model.pth"
model3_path = r"D:\dh\Dacon\Xception\convnextv2_best_model.pth"

model1_name = 'resnet101'
model2_name = 'xception65'
model3_name = 'convnextv2_tiny'

# 모델 1: resnet101
resnet_model = get_model(model1_name, num_classes=config.num_classes, pretrained=False)
resnet_model.load_state_dict(torch.load(model1_path, map_location=device))
resnet_model = resnet_model.to(device)
resnet_model.eval()

# 모델 2: xception65
xception_model = get_model(model2_name, num_classes=config.num_classes, pretrained=False)
xception_model.load_state_dict(torch.load(model2_path, map_location=device))
xception_model = xception_model.to(device)
xception_model.eval()

# 모델 3: convnextv2
convnext_model = ConvNeXtV2WithDropout(num_classes=config.num_classes, dropout_p=0.3)
convnext_model.load_state_dict(torch.load(model3_path, map_location=device))
convnext_model = convnext_model.to(device)
convnext_model.eval()

# 🔍 추론 및 앙상블 (soft voting)
all_preds = []

with torch.no_grad():
    for images in tqdm(test_loader, desc="🔍 Inference", ncols=100):
        images = images.to(device)

        out1 = resnet_model(images)
        out2 = xception_model(images)
        out3 = convnext_model(images)

        prob1 = F.softmax(out1, dim=1)
        prob2 = F.softmax(out2, dim=1)
        prob3 = F.softmax(out3, dim=1)

        # 3개의 모델의 확률을 평균하여 앙상블
        ensemble_prob = (prob1 + prob2 + prob3) / 3
        preds = ensemble_prob.argmax(dim=1)

        # 클래스 인덱스를 라벨명으로 변환
        preds = le.inverse_transform(preds.cpu().numpy())
        all_preds.extend(preds)

# 📄 저장
submit = pd.read_csv('../data/sample_submission.csv')
submit['rock_type'] = all_preds
submit.to_csv('../output/ensemble_submit.csv', index=False)
print("✅ ensemble inference finished → ensemble_submit.csv")
