import torch
import torch.nn as nn
from torchvision.models import resnet50


class ResNet50Encoder(nn.Module):
    def __init__(self, n_style, style_dim=512, mean_latent=None, pretrained=False):
        super().__init__()
        self.n_style = n_style
        self.style_dim = style_dim
        # copy all the modules from an existing resnet
        model_tmp = resnet50(pretrained=pretrained)
        self.conv1 = model_tmp.conv1
        self.bn1 = model_tmp.bn1
        self.relu = model_tmp.relu
        self.maxpool = model_tmp.maxpool

        self.layer1 = model_tmp.layer1
        self.layer2 = model_tmp.layer2
        self.layer3 = model_tmp.layer3
        self.layer4 = model_tmp.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(model_tmp.fc.in_features, n_style * style_dim)

        # register a mean buffer
        self.register_buffer('mean_latent', torch.rand(style_dim))
        if mean_latent is not None:  # taken from
            self.mean_latent.data.copy_(mean_latent)

    def forward(self, x):
        assert x.shape[-1] == x.shape[-2] == 256  # only accept input with resolution 256x256
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        x = x.view(x.shape[0], self.n_style, self.style_dim) + self.mean_latent.view(1, 1, -1).detach()

        return x  # shape: n, n_style, style_dim