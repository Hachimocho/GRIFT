#  Thanks to https://blog.floydhub.com/gru-with-pytorch/ for some of the recurrent code!
import torch.nn as nn
from torch.hub import load

# list('zhanghang1989/ResNeSt', force_reload=True)

class ModelOut(nn.Module):
    def __init__(self, pretrained: bool =False, finetune : bool =False, exclude_top : bool = False,
                    output_classes: int = 3, classification_strategy: str = 'categorical', configuration: str = 'default'
        ):
        super(ModelOut, self).__init__()
        resnest50_base = load('zhanghang1989/ResNeSt', 'resnest50', pretrained=pretrained)
        self.model = resnest50_base
        self.in_features = resnest50_base.fc.in_features
        self.out_features = output_classes if classification_strategy == 'categorical' else 1
        self.classification_strategy = classification_strategy
        if finetune:
            for param in self.model.parameters():
                param.requires_grad = False
        
        '''
        # view layer names
        for name, layer, in resnest50_base.named_modules():
            print(name, layer)
        # fc Linear(in_features=2048, out_features=1000, bias=True)
        '''
        
        # edit the last layer [fc]
        #if exclude_top:
            #self.model.fc = Identity()
        #else:
        self.g1 = nn.GRU(self.in_features, 1024, 10, batch_first=True),
        self.l1 = nn.Linear(
            1024, 512
        ),
        self.b1 = nn.BatchNorm1d(512),
        self.r1 = nn.ReLU(),
        self.d1 = nn.Dropout(0.4),
        self.l2 = nn.Linear(512, self.out_features)
        self.s1 = nn.Softmax(dim=1)
            #if classification_strategy == 'categorical':
                #fc.add_module(module=nn.Softmax(dim=1), name="Categorical_Softmax")
            
            #self.model.fc = fc
    
    def forward(self, x, h):
        # forward pass
        x, h = self.g1(x, h)
        x = self.l2(self.d1(self.r1(self.b1(self.l1(x[:, -1])))))
        if self.classification_strategy == 'categorical':
            x = self.s1(x)
        return x, h
    
    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x, h):
        return x, h

def main():
    from torchsummary import summary
    model = ModelOut(False, False)
    summary(model.cuda(), (3,224,224))

if __name__ == "__main__":
    main()