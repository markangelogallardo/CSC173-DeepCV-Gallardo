import torch.nn as nn
from torchvision import models

def get_model(model_name, num_classes=50, pretrained=True):
    # 'DEFAULT' fetches the best available weights (IMAGENET1K_V1 or V2)
    weights = 'DEFAULT' if pretrained else None
    
    if model_name == 'resnet18':
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'vgg16':
        model = models.vgg16(weights=weights)
        # VGG classifier: (0): Linear ... (6): Linear
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=weights)
        # MobileNetV2 classifier: (0): Dropout, (1): Linear
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    elif model_name == 'mobilenet_v3':
        # FIX: Must specify 'large' or 'small'. Defaulting to large.
        model = models.mobilenet_v3_large(weights=weights)
        # FIX: MobileNetV3 classifier is: Linear -> Hardswish -> Dropout -> Linear
        # The final layer is at index 3 (or -1)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    elif model_name == 'shufflenet_v2':
        model = models.shufflenet_v2_x1_0(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=weights)
        # EfficientNet classifier: (0): Dropout, (1): Linear
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == 'regnet_y_800mf':
        model = models.regnet_y_800mf(weights=weights)
        # FIX: RegNet uses 'fc', not 'classifier'
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    else:
        raise ValueError(f"Model {model_name} not supported.")
        
    return model