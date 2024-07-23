import torch.nn as nn
from torch.hub import load
from facenet_pytorch import InceptionResnetV1

# list('zhanghang1989/ResNeSt', force_reload=True)


class ModelOut(nn.Module):
    def __init__(self, pretrained: bool = False, finetune: bool = False, exclude_top: bool = False,
                 output_classes: int = 3, classification_strategy: str = 'categorical', configuration: str = 'default'
                 ):
        super(ModelOut, self).__init__()
        model = InceptionResnetV1(pretrained='vggface2', classify=True)
        # resnest50_base = load('zhanghang1989/ResNeSt',
        # 'resnest50', pretrained=pretrained)
        # self.model = resnest50_base
        self.model = model
        for name, layer in model.named_modules():
            print(name, layer)
        # print(model.named_parameters())
        self.in_features = model.logits.in_features
        self.out_features = output_classes if classification_strategy == 'categorical' else 1

        for param in self.model.parameters():
            param.requires_grad = False

        '''
        # view layer names
        for name, layer, in resnest50_base.named_modules():
            print(name, layer)
        # fc Linear(in_features=2048, out_features=1000, bias=True)
        '''

        # edit the last layer [fc]
        if exclude_top:
            self.model.logits = Identity()
        else:
            logits = None
            if configuration == 'default':
                logits = nn.Sequential(
                    nn.Linear(
                        self.in_features, 1024
                    ),
                    nn.Dropout(0.4),
                    nn.Linear(1024, self.out_features),
                )
            elif configuration == 'bn':
                logits = nn.Sequential(
                    nn.Linear(
                        self.in_features, 1024
                    ),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(
                        1024, 512
                    ),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, self.out_features)
                )
            elif configuration == 'elu':
                logits = nn.Sequential(
                    nn.Linear(
                        self.in_features, 1024
                    ),
                    nn.ELU(),
                    nn.Dropout(0.4),
                    nn.Linear(1024, self.out_features)
                )
            if classification_strategy == 'categorical':
                logits.add_module(module=nn.Softmax(dim=1),
                                  name="Categorical_Softmax")

            self.model.logits = logits
            # for name, layer in model.named_modules():
            #     print(name, layer)

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
