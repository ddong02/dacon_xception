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
from torch.utils.data import Subset

def main():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # 데이터프레임 생성
        train_df, val_df, test_df, class_to_idx = load_dataframe(
            config.train_dir, config.test_dir, config.test_csv_path,
            test_size=config.test_size, shuffle=config.data_shuffle
        )

        # 테스트를 위해 100 개만 추출
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

        # 손실 함수와 옵티마이저
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        plotter = Plot_graph(save_path="../output/graphs/output_graph.png")

        # 학습
        best_val_acc = 0.0
        best_epoch = -1

        # 테스트를 위해 epochs 과 batch_size 조절
        config.n_epochs = 10
        config.batch_size = 8

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
            # 테스트용이므로 저장 생략
            # save_checkpoint(model, epoch_save_path)

            # 최고 성능 모델 저장
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                # 테스트용이므로 저장 생략
                # save_checkpoint(model, config.model_save_path)
                print(f"✅ Best model saved with acc {best_val_acc:.4f}")

        print(f"\nTraining complete. Best model was from epoch {best_epoch} with acc {best_val_acc:.4f}")
        # 테스트용이므로 저장 생략
        # plotter.save_and_close()    # 저장 후 닫기

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