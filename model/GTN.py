import torch
import torch.nn as nn
import math
import torchvision.models as models

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

class GTN(nn.Module):
    def __init__(self, out_planes, pretrained = False, reduction = 16):
        super(GTN, self).__init__()
        ResNet = models.resnet50(pretrained=pretrained)
        self.feature_extract = nn.Sequential(
            ResNet.conv1,
            ResNet.bn1,
            ResNet.relu,
            ResNet.maxpool,
            ResNet.layer1,
            ResNet.layer2,
            ResNet.layer3
            )
        self.auxlayer = nn.Sequential(
            nn.AvgPool2d(14),
            Flatten(),
            nn.Linear(1024, out_planes, bias=False)
            )
        self.layer4 = nn.Sequential(
            ResNet.layer4,
            ResNet.avgpool,
            Flatten()
            )
        self.fc = nn.Sequential(
            nn.Linear(2048, 2048//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),
            nn.Linear(2048//reduction, 2048, bias=False),
            nn.Sigmoid()
            )
        self.classify = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048, out_planes, bias=False)
            )

    def forward(self, x):
        x = self.feature_extract(x)
        aux = self.auxlayer(x)
        x = self.layer4(x)

        residual = x
        weight = self.fc(x)
        out = self.classify(weight*residual)
        return out, aux

if __name__ == '__main__':
    img = torch.randn(2, 3, 224, 224)
    model = GTN(4)
    out, aux = model(img)
    print(out.size())
