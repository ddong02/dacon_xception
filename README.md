![](https://velog.velcdn.com/images/ddong02/post/a82562a9-b387-4bf2-a89d-9cbf103bb5c6/image.png)

---

**사용 모델: xception65**

## 기본 코드

- **특이사항**
데이터 증강 부분만 추가했을 때 최고점수 0.82646
학습률 스케줄러 없음
클래스 가중치 없음
시드 고정 안함 (고정해도 위의 점수와 비슷함)

---

### `config.py`

```python
# my_config.py

import cv2
from albumentations import (
    Compose, HorizontalFlip, ShiftScaleRotate, Blur, CoarseDropout, 
    RandomRain, CLAHE, ColorJitter, RandomBrightnessContrast, OneOf,
    PadIfNeeded, Resize, Normalize
)
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform

class PadSquare(ImageOnlyTransform):
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
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.value)
        return image

    def get_transform_init_args_names(self):
        return ("border_mode", "value")

class Config:
    # 데이터 경로
    train_dir = '../data/train'
    test_dir = '../data/test'
    test_csv_path = '../data/test.csv'

    # 데이터 설정
    test_size = 0.25
    data_shuffle = True
    image_size = (224, 224)
    batch_size = 32

    # 학습 설정
    n_epochs = 50
    learning_rate = 1e-4
    num_classes = 7
    model_name = 'xception65'

    # 파일 저장
    model_save_path = '../output/best_model.pth'

    # Augmentation 설정
    train_augmentor = Compose([
        PadSquare(value=(0, 0, 0)),
        Resize(image_size[1], image_size[0]),
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=(0.2, 0.1), scale_limit=(0.7, 1.0), rotate_limit=90, p=0.5),
        Blur(blur_limit=8, p=0.4),
        CoarseDropout(max_holes=30, max_height=16, max_width=16, p=0.2),
        OneOf([
            ColorJitter(0.2, 0.2, 0.2, 0.2, p=1.0),
            RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0)
        ], p=0.4),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    test_augmentor = Compose([
        PadSquare(value=(0, 0, 0)),
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
```

---

### `dataset.py`

```python
# my_dataset.py

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
```

---

### inference.py

```python
# my_inference

import torch
from tqdm import tqdm

def inference(model, test_loader, device, label_encoder=None):

    print('\n')
    print('-'*30)
    print('inference starts ...\n')

    model = model.to(device)
    model.eval()
    preds = []

    with torch.no_grad():
        for imgs in tqdm(test_loader, desc="Inference"):
            imgs = imgs.to(device).float()
            outputs = model(imgs)
            pred_labels = outputs.argmax(dim=1).detach().cpu().numpy().tolist()
            preds.extend(pred_labels)

    if label_encoder:
        preds = label_encoder.inverse_transform(preds)

    print("✅ inference finished → baseline_submit.csv")

    return preds
```

---

### `model.py`

```python
# my_model.py

import torch.nn as nn
import torchvision.models as models

try:
    import timm
except ImportError:
    print("Warning: 'timm' not installed. Install it via 'pip install timm' to use models like Xception.")

def get_model(model_name='convnext_base', num_classes=7, pretrained=True):
    model_name = model_name.lower()

    if model_name == 'convnext_base':
        model = models.convnext_base(pretrained=pretrained)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)

    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == 'xception':
        model = timm.create_model('legacy_xception', pretrained=pretrained, num_classes=num_classes)

    elif model_name == 'xception65':
        model = timm.create_model('xception65', pretrained=pretrained, num_classes=num_classes)

    else:
        raise ValueError(f"Model '{model_name}' is not supported.")

    return model
```

---

### `plot.py`

```python
# my_plot_util.py

import matplotlib.pyplot as plt
import os

class Plot_graph:
    def __init__(self, save_path="./save_path"):
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 2, figsize=(15, 5))
        self.save_path = save_path

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def update(self, epoch, train_loss, val_loss, train_acc, val_acc):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)

        epochs = range(1, epoch + 2)

        # Loss plot
        self.axes[0].clear()
        self.axes[0].plot(epochs, self.train_losses, label="Train Loss", marker="o")
        self.axes[0].plot(epochs, self.val_losses, label="Val Loss", marker="o")
        self.axes[0].set_title("Loss")
        self.axes[0].set_xlabel("Epoch")
        self.axes[0].set_ylabel("Loss")
        self.axes[0].legend()
        self.axes[0].grid(True)

        # Accuracy plot
        self.axes[1].clear()
        self.axes[1].plot(epochs, self.train_accuracies, label="Train Accuracy", marker="o")
        self.axes[1].plot(epochs, self.val_accuracies, label="Val Accuracy", marker="o")
        self.axes[1].set_title("Accuracy")
        self.axes[1].set_xlabel("Epoch")
        self.axes[1].set_ylabel("Accuracy")
        self.axes[1].legend()
        self.axes[1].grid(True)

        plt.tight_layout()
        plt.pause(0.1)

        save_name = os.path.join(
        os.path.dirname(self.save_path),
        f"epoch_{epoch+1:03d}.png"
        )
        self.fig.savefig(save_name)

    def save_and_close(self, interrupt=False):
        path = self.save_path.replace(".png", "_interrupt.png") if interrupt else self.save_path
        plt.ioff()
        plt.savefig(path)
        plt.close(self.fig)
```

---


### `train.py`

```python
# my_train.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import os

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0

    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)

    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()

            # f1 score 계산을 위한 누적
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(classification_report(all_labels, all_preds))

    return avg_loss, accuracy, f1

def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
```

---

### main.py

```python
# main.py

import torch
import pandas as pd
import os
from torch.utils.data import DataLoader
from sklearn import preprocessing
from my_dataset import load_dataframe, StoneDataset
from my_config import config
from my_model import get_model
from my_train import train_one_epoch, validate, save_checkpoint
from my_plot_util import Plot_graph
from my_inference import inference

def main():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # 데이터프레임 생성
        train_df, val_df, test_df, class_to_idx = load_dataframe(
            config.train_dir, config.test_dir, config.test_csv_path,
            test_size=config.test_size, shuffle=config.data_shuffle
        )
    
        # Dataset 및 DataLoader 정의
        train_dataset = StoneDataset(train_df, image_size=config.image_size, transform=config.train_augmentor)
        val_dataset = StoneDataset(val_df, image_size=config.image_size, transform=config.val_augmentor)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

        # 모델 생성
        model = get_model(config.model_name, num_classes=config.num_classes, pretrained=True)
        model = model.to(device)

        # 손실 함수와 옵티마이저
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        plotter = Plot_graph(save_path="../output/graphs/output_graph.png")

        # 학습
        best_val_acc = 0.0
        best_epoch = -1
        for epoch in range(config.n_epochs):
            print(f"\nEpoch {epoch+1}/{config.n_epochs}")

            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

            val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
            print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
            print(f">>> F1 score (macro) : {val_f1:.4f}")   # f1 score 출력

            # plot 업데이트
            plotter.update(epoch, train_loss, val_loss, train_acc, val_acc)

            # 각 에폭마다 모델 저장
            epoch_save_path = f"../output/models/model_epoch{epoch+1:02d}.pth"
            save_checkpoint(model, epoch_save_path)

            # 최고 성능 모델 저장
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                save_checkpoint(model, config.model_save_path)
                print(f"✅ Best model saved with acc {best_val_acc:.4f}")

        print(f"\nTraining complete. Best model was from epoch {best_epoch} with acc {best_val_acc:.4f}")
        plotter.save_and_close()    # 저장 후 닫기

        test_dataset = StoneDataset(test_df, image_size=config.image_size, transform=config.test_augmentor, is_test=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

        le = preprocessing.LabelEncoder()
        le.fit(train_df['label'])

        infer_model = get_model(config.model_name, num_classes=config.num_classes, pretrained=False)
        state_dict = torch.load(config.model_save_path, map_location=device)
        infer_model.load_state_dict(state_dict)

        infer_model = infer_model.to(device)

        preds = inference(model=infer_model, test_loader=test_loader, device=device, label_encoder=le)

        submit = pd.read_csv('../data/sample_submission.csv')
        submit['rock_type'] = preds
        submit.to_csv('../output/baseline_submit.csv', index=False)

    except KeyboardInterrupt:
        print('\n🛑 Keyboardinterrupt: Training interrupted by user.')
        plotter.save_and_close(interrupt=True)

if __name__ == "__main__":
    main()
```

---

## 앙상블 기법

**추론 과정에서 softmax 확률을 계산해 평균내는 Soft Voting 방식**
`resnet101`, `xception65`, `convnextv2_tiny` 3개의 `bestmodel.pth` 파일을 이용
**최고 점수 0.85731**

---

### ensemble.py

```python
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
```

---

## 느낀점

1. 초반에 Baseline 코드 분석하고 시작하기
2. Baseline 코드에서 크게 달라지지 않도록 코드 수정 (호환성을 위해)
3. Dataset 분석해서 모델 학습 방법, 증강 방법 생각해보기
4. 클래스 가중치, 학습률 스케줄러, fine tuning 등 여러 가지 방법 하나씩 해보면서 성능 차이 파악하기
5. wandb 같은 외부 툴 이용해서 훈련 결과 관리하기
6. 
