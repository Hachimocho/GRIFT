import torch.nn as nn
from torch.hub import load

# list('zhanghang1989/ResNeSt', force_reload=True)


class ModelOut(nn.Module):
    def __init__(self, pretrained: bool = False, finetune: bool = False, exclude_top: bool = False,
                 output_classes: int = 3, classification_strategy: str = 'categorical', configuration: str = 'default'
                 ):
        super(ModelOut, self).__init__()
        squeezenet_base = load('pytorch/vision:v0.10.0',
                               'squeezenet1_1', pretrained=pretrained)
        self.model = squeezenet_base
        for name, layer, in squeezenet_base.named_modules():
            print(name, layer)
        # self.in_features = squeezenet_base.fc.in_features
        # self.out_features = output_classes if classification_strategy == 'categorical' else 1

        for param in self.model.parameters():
            param.requires_grad = False

        '''
        # view layer names
        for name, layer, in resnest50_base.named_modules():
            print(name, layer)
        # fc Linear(in_features=2048, out_features=1000, bias=True)
        '''

        # edit the last layer [fc]
        # if exclude_top:
        #     self.model.fc = Identity()
        # else:
        #     fc = None
        #     if configuration == 'default':
        #         fc = nn.Sequential(
        #             nn.Linear(
        #                 self.in_features, 1024
        #             ),
        #             nn.Dropout(0.4),
        #             nn.Linear(1024, self.out_features),
        #         )
        #     elif configuration == 'bn':
        #         fc = nn.Sequential(
        #             nn.Linear(
        #                 self.in_features, 1024
        #             ),
        #             nn.BatchNorm1d(1024),
        #             nn.ReLU(),
        #             nn.Dropout(0.5),
        #             nn.Linear(
        #                 1024, 512
        #             ),
        #             nn.BatchNorm1d(512),
        #             nn.ReLU(),
        #             nn.Dropout(0.5),
        #             nn.Linear(512, self.out_features)
        #         )
        #     elif configuration == 'elu':
        #         fc = nn.Sequential(
        #             nn.Linear(
        #                 self.in_features, 1024
        #             ),
        #             nn.ELU(),
        #             nn.Dropout(0.4),
        #             nn.Linear(1024, self.out_features)
        #         )
        #     if classification_strategy == 'categorical':
        #         fc.add_module(module=nn.Softmax(dim=1),
        #                       name="Categorical_Softmax")

        #     self.model.fc = fc
        self.model.classifier[1] = nn.Conv2d(
            512, output_classes, kernel_size=(1, 1), stride=(1, 1))

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
