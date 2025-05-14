# model.py
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
        model = timm.create_model('xception', pretrained=pretrained, num_classes=num_classes)

    else:
        raise ValueError(f"Model '{model_name}' is not supported.")

    return model
