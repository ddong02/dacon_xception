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