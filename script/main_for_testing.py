# test main.py

import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn import preprocessing
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from my_dataset import load_dataframe, StoneDataset
from my_config import config
from my_model import get_model
from my_train import train_one_epoch, validate, save_checkpoint
from my_plot_util import Plot_graph
from my_inference import inference
from early_stopping import EarlyStopping  # early stopping 모듈

def main():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # 데이터프레임 생성
        train_df, val_df, test_df, class_to_idx = load_dataframe(
            config.train_dir, config.test_dir, config.test_csv_path,
            test_size=config.test_size, shuffle=config.data_shuffle
        )

        # 테스트를 위해 100개로 축소 (원본 코드 유지)
        train_df = train_df.iloc[:100].reset_index(drop=True)
        val_df = val_df.iloc[:100].reset_index(drop=True)
        test_df = test_df.iloc[:100].reset_index(drop=True)

        # Dataset 및 DataLoader 정의
        train_dataset = StoneDataset(train_df, image_size=config.image_size, transform=config.train_augmentor)
        val_dataset = StoneDataset(val_df, image_size=config.image_size, transform=config.val_augmentor)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

        # 모델 생성
        model = get_model(config.model_name, num_classes=config.num_classes, pretrained=True)
        model = model.to(device)

        # class weight 적용
        labels = train_df['label_idx'].values
        # class_labels = np.unique(labels)
        # test를 위해서 0~6 클래스를 강제로 생성 (실제 훈련은 위의 코드로)
        class_labels = np.arange(config.num_classes)
        class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=labels)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

        # 손실 함수 및 옵티마이저
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        plotter = Plot_graph(save_path="../output/graphs/output_graph.png")

        # early stopping 설정
        early_stopping = EarlyStopping(patience=5, verbose=True, delta=1e-3,
                                       path='../output/ealry_stopping_model.pth')

        best_val_acc = 0.0
        best_val_f1 = 0.0
        best_epoch = -1
        best_f1_epoch = -1

        for epoch in range(config.n_epochs):
            print(f"\nEpoch {epoch+1}/{config.n_epochs}")

            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print()

            val_loss, val_acc, val_f1, all_labels, all_preds = validate(model, val_loader, criterion, device)
            print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
            print(f">>> F1 score (macro) : {val_f1:.4f}")

            print()
            print('-' * 60)
            print(classification_report(all_labels,
                                        all_preds,
                                        target_names=[ "Andesite",
                                                        "Basalt",
                                                        "Etc",
                                                        "Gneiss",
                                                        "Granite",
                                                        "Mud_Sandstone",
                                                        "Weathered_Rock" ],
                                        zero_division=0))
            print('-' * 60)            

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

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_f1_epoch = epoch + 1

            # early stopping 체크
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print()
                print('-' * 30)
                print("Early stopping triggered")
                print(f"Last epoch was {epoch+1}\n")
                print('-' * 30)
                break

        print(f"\nTraining complete. Best model was from epoch {best_epoch} with acc {best_val_acc:.4f}")
        print(f"Best F1 score was {best_val_f1:.4f} at epoch {best_f1_epoch}")
        plotter.save_and_close()

        # 테스트 수행
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
