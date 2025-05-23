# inference_only.py

import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn import preprocessing
from my_config import config
from my_dataset import load_dataframe, StoneDataset
from my_model import get_model
from my_inference import inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Inference Mode] Using device: {device}")

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

# 모델 로드
model_path = config.model_save_path

model = get_model(config.model_name, num_classes=config.num_classes, pretrained=False)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)

# 추론
preds = inference(model=model, test_loader=test_loader, device=device, label_encoder=le)

# 저장
submit = pd.read_csv('../data/sample_submission.csv')
submit['rock_type'] = preds
submit.to_csv('../output/baseline_submit.csv', index=False)
print("✅ inference finished → baseline_submit.csv")