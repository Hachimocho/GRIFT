from distutils.command.config import config
import torch.nn as nn
from torch.hub import load
import torch
import pickle

effnetargs = {
"pretrained": True,
"finetune": False,
"output_classes": 2,
"classification_strategy": 'binary',
"configuration": "bn",
"exclude_top": False}

class ModelOut(nn.Module):

    def __init__(self, pretrained: bool =False, finetune : bool =False, exclude_top : bool = False,
                    output_classes: int = 3, classification_strategy: str = 'categorical', configuration: str = 'default'
        ):
        super(ModelOut, self).__init__()
        efficientnet_base = load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=pretrained)
        self.model = efficientnet_base
        self.in_features = efficientnet_base.classifier.fc.in_features
        self.out_features = output_classes if classification_strategy == 'categorical' else 1
        
        # freeze the parameters so that the gradients are not computed 
        if finetune:
            for param in self.model.parameters():
                param.requires_grad = False        
        
        '''
        # view layer names
        for name, layer in efficientnet_base.named_modules():
            print(name, layer)
        # classifier.fc Linear(in_features=1792, out_features=1000, bias=True)
        '''

        # self.model.classifier.pooling = GlobalAvgPool2d()
        self.model.classifier.pooling = nn.AdaptiveMaxPool2d(1) # this fixed the network surprisingly

        if exclude_top:
            self.model.fc = Identity()
        else:
            fc = None
            if configuration == 'default':
                fc = nn.Sequential(
                    nn.Linear(
                        self.in_features, 1024
                    ),
                    nn.Dropout(0.4),
                    nn.Linear(1024, self.out_features),
                )
            elif configuration == 'bn':
                fc = nn.Sequential(
                    nn.Linear(
                        self.in_features, 1024
                    ),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(1024, self.out_features)
                )
            elif configuration == 'rlbn':
                fc = nn.Sequential(
                    nn.Linear(
                        self.in_features, 1024
                    ),
                    nn.ReLU(),
                    nn.BatchNorm1d(1024),
                    nn.Dropout(0.4),
                    nn.Linear(1024, self.out_features)
                )
            elif configuration == 'elu':
                fc = nn.Sequential(
                    nn.Linear(
                        self.in_features, 1024
                    ),
                    nn.ELU(),
                    nn.Dropout(0.4),
                    nn.Linear(1024, self.out_features)
                )
            if classification_strategy == 'categorical':
                fc.add_module(module=nn.Softmax(dim=1), name="Categorical_Softmax")
            
            self.model.classifier.fc = fc
            # # edit the last layer [fc]
            # self.model.classifier.fc = nn.Sequential(
            #     nn.Linear(
            #         self.in_features, 1024
            #     ),
            #     nn.Dropout(0.4),
            #     nn.Linear(1024, self.out_features),
            #     nn.Softmax(dim=1)
            # ) if classification_strategy == 'categorical' else nn.Sequential(
            #     nn.Linear(
            #         self.in_features, 1024
            #     ),
            #     nn.BatchNorm1d(1024),
            #     nn.ReLU(),
            #     nn.Dropout(0.4),
            #     nn.Linear(1024, self.out_features)
            # )

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
    summary(model.cuda(), (3,224,224))

if __name__ == "__main__":
    main()
    
def predict(data):
    with open('/home/brg2890/major/bryce_python_workspace/deepfake-detection-models_altrepo/weights/binary_trn_t_tst_t_aug_t_auto_f_new_bn_v2/config.pk', mode='rb') as file:
        config = pickle.load(file)
    model = ModelOut(**config)
    model.eval()
    prediction = model(data)
    return prediction