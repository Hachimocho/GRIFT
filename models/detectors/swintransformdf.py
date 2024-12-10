import torch.nn as nn
from torch.hub import load
import torchvision.models as models

# list('zhanghang1989/ResNeSt', force_reload=True)


class ModelOut(nn.Module):
    def __init__(self, pretrained: bool = False, finetune: bool = False, exclude_top: bool = False,
                 output_classes: int = 3, classification_strategy: str = 'categorical', configuration: str = 'default'):
        super(ModelOut, self).__init__()
        
        # Load pretrained Swin Transformer
        self.model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT if pretrained else None)
        self.in_features = self.model.head.in_features
        self.out_features = output_classes if classification_strategy == 'categorical' else 1
        
        # Modify the head based on configuration
        if exclude_top:
            self.model.head = Identity()
        else:
            head = None
            if configuration == 'default':
                head = nn.Sequential(
                    nn.Linear(self.in_features, 1024),
                    nn.Dropout(0.4),
                    nn.Linear(1024, self.out_features),
                )
            elif configuration == 'bn':
                head = nn.Sequential(
                    nn.Linear(self.in_features, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(1024, self.out_features)
                )
            elif configuration == 'elu':
                head = nn.Sequential(
                    nn.Linear(self.in_features, 1024),
                    nn.ELU(),
                    nn.Dropout(0.4),
                    nn.Linear(1024, self.out_features)
                )
            
            if classification_strategy == 'categorical':
                head.add_module(module=nn.Softmax(dim=1), name="Categorical_Softmax")
            
            self.model.head = head
        
        # Freeze feature extraction layers if fine-tuning
        if finetune:
            for name, param in self.model.named_parameters():
                if 'head' not in name:  # Don't freeze head layers
                    param.requires_grad = False

    def forward(self, x):
        # forward pass
        return self.model(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def main():
    from torchsummary import summary
    model = ModelOut(False, False)
    summary(model.cuda(), (3, 224, 224))


if __name__ == "__main__":
    main()
