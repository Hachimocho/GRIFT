import torch.nn as nn
import torch.hub
import torch
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        try:
            # `torch.hub.load` resolved at call time, not bound at import. `from torch.hub
            # import load` captured whatever the attribute was when this module was first
            # imported -- so the test tier's network block either leaked in permanently (if
            # the module was imported while it was active) or was bypassed entirely (if it
            # was imported before), depending only on import order.
            efficientnet_base = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=pretrained)
            logger.info(f"Successfully loaded EfficientNet model")
        except Exception as e:
            logger.error(f"Error loading EfficientNet model: {e}")
            raise
            
        self.model = efficientnet_base
        self.in_features = efficientnet_base.classifier.fc.in_features
        self.out_features = output_classes if classification_strategy == 'categorical' else 1
        
        # freeze the parameters so that the gradients are not computed 
        if finetune:
            # Only freeze the feature extraction layers
            for name, param in self.model.named_parameters():
                if 'classifier' not in name:  # Don't freeze classifier layers
                    param.requires_grad = False        
        
        '''
        # view layer names
        for name, layer in efficientnet_base.named_modules():
            print(name, layer)
        # classifier.fc Linear(in_features=1792, out_features=1000, bias=True)
        '''

        # The backbone's native pooling is AdaptiveAvgPool2d(1), and it is left in place.
        #
        # This line used to unconditionally replace it with AdaptiveMaxPool2d(1), commented
        # "this fixed the network surprisingly". Measured on the pretrained b4's real
        # 1792x8x8 feature map, max pooling inflates the activation scale by **42.7x**
        # (mean 0.088 -> 3.756, std 0.099 -> 1.833). That output feeds a freshly initialized
        # Linear(1792, 1024) whose default init assumes roughly unit-scale inputs, so the
        # pre-activations saturate and the gradients vanish -- which is consistent with
        # effnetdf scoring AUROC 0.50 and 0.50 in two consecutive sweeps, at chance, while
        # detectors sharing the same head shape reached 0.71-0.84.
        #
        # Kept reachable, because the original comment suggests it helped in some earlier
        # configuration and this has not been A/B'd on the real dataset -- but off by
        # default, since it is inconsistent with how the pretrained weights were trained.
        if configuration == 'maxpool':
            self.model.classifier.pooling = nn.AdaptiveMaxPool2d(1)

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