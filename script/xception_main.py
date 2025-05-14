# main.py
import torch
from torch.utils.data import DataLoader
from my_dataset import load_dataframe, StoneDataset
from my_config import config
from my_model import get_model
from my_train import train_one_epoch, validate, save_checkpoint

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
        train_dataset = StoneDataset(train_df, image_size=config.image_size)

        val_dataset = StoneDataset(val_df, image_size=config.image_size, transform=config.val_augmentor)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

        # 모델 생성
        model = get_model(config.model_name, num_classes=config.num_classes, pretrained=True)
        model = model.to(device)

        # 손실 함수와 옵티마이저
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        # 학습
        best_val_acc = 0.0
        for epoch in range(config.n_epochs):
            print(f"\nEpoch {epoch+1}/{config.n_epochs}")

            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

            val_loss, val_acc = validate(model, val_loader, criterion, device)
            print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

            # 각 에폭마다 모델 저장
            epoch_save_path = f"../output/model_epoch{epoch+1:02d}.pth"
            save_checkpoint(model, epoch_save_path)

            # 최고 성능 모델 저장
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(model, config.model_save_path)
                print(f"✅ Best model saved with acc {best_val_acc:.4f}")
    except:
        print('\n🛑 Keyboardinterrupt: Training interrupted by user.')

if __name__ == "__main__":
    main()
