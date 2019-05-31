import torch
import torch.nn as nn
import math
import torchvision.models as models

class GTN(nn.Module):
    def __init__(self, out_planes, pretrained=False):
        super(GTN, self).__init__()
        self.model = nn.Sequential(models.resnet50(pretrained=pretrained)[:-1]) 
        self.auxpool = nn.AvgPool2d(14)
        self.aux = nn.Linear(256 * 4, num_classes)
        self.efc1 = nn.Linear(512 * 4, 512 * 4//reduction)
        # self.ebn1 = nn.BatchNorm2d(1000)
        self.erelu1 = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(p=0.7)
        self.efc2 = nn.Linear(512 * 4//reduction, 512 * 4)
        self.en2 = nn.Sigmoid()
        self.dp2 = nn.Dropout(p=0.5)
        # self.ebn2 = nn.BatchNorm2d(512 * block.expansion)
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        aux = self.auxpool(x)
        aux = aux.view(aux.size(0), -1)
        aux = self.aux(aux)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        residual = x
        out = self.efc1(x)
        # out = self.ebn1(out)
        out = self.erelu1(out)
        out = self.dp(out)
        out = self.efc2(out)
        out = self.en2(out)
        weight = self.dp2(out)
        out = weight.clone()*residual
        out1 = self.fc(out)
        return weight, out1

