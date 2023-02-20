import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from itertools import chain



import torch.nn as nn

def convert_relu_to_swish(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.SiLU())
        elif isinstance(child, nn.LeakyReLU):
            setattr(model, child_name, nn.SiLU())
        else:
            convert_relu_to_swish(child)
            
            
            
def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()


class PSPModule(nn.Module):
    # In the original inmplementation they use precise RoI pooling
    # Instead of using adaptative average pooling
    def __init__(self, in_channels, bin_sizes=[1, 2, 4, 6]):
        super(PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s)
                                     for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+(out_channels * len(bin_sizes)), in_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear',
                                       align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class ResNet(nn.Module):
    def __init__(self, num_feature, backbone='resnet34', pretrained=True):
        super(ResNet, self).__init__()
        model = getattr(models, backbone)(weights=None)

        # self.initial = nn.Sequential(nn.Conv2d(num_feature, 64, 1), *list(model.children())[1:4])
        self.initial = nn.Sequential(nn.Conv2d(num_feature, 64, 1), *list(model.children())[1:3])
        
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        s3, s4, d3, d4 = (2, 1, 1, 2)


        for n, m in self.layer4.named_modules():
            if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'conv2' in n:
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'downsample.0' in n:
                m.stride = (s4, s4)

    def forward(self, x):
        x = self.initial(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x1, x2, x3, x4]


def up_and_add(x, y):
    return F.interpolate(x, size=(y.size(2), y.size(3)), mode='bilinear', align_corners=True) + y


class FPN_fuse(nn.Module):
    def __init__(self, feature_channels=[256, 512, 1024, 2048], fpn_out=256):
        super(FPN_fuse, self).__init__()
        assert feature_channels[0] == fpn_out
        self.conv1x1 = nn.ModuleList([nn.Conv2d(ft_size, fpn_out, kernel_size=1)
                                      for ft_size in feature_channels[1:]])
        self.smooth_conv = nn.ModuleList([nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)]
                                         * (len(feature_channels)-1))
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels)*fpn_out, fpn_out,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):

        features[1:] = [conv1x1(feature) for feature,
                        conv1x1 in zip(features[1:], self.conv1x1)]
        P = [up_and_add(features[i], features[i-1])
             for i in reversed(range(1, len(features)))]
        P = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, P)]
        P = list(reversed(P))
        P.append(features[-1])  # P = [P1, P2, P3, P4]
        H, W = P[0].size(2), P[0].size(3)
        P[1:] = [F.interpolate(feature, size=(
            H, W), mode='bilinear', align_corners=True) for feature in P[1:]]

        x = self.conv_fusion(torch.cat((P), dim=1))
        return x


class UperNet(nn.Module):
    # Implementing only the object path
    def __init__(self, num_features, out_channel, backbone='resnet34', pretrained=False, fpn_out=64):
        super(UperNet, self).__init__()

        if backbone == 'resnet34' or backbone == 'resnet18':
            feature_channels = [64, 128, 256, 512]
        else:
            feature_channels = [256, 512, 1024, 2048]
        self.feature_channels = feature_channels
        self.backbone = ResNet(num_features, backbone, pretrained=pretrained)
        self.PPN = PSPModule(feature_channels[-1])
        self.FPN = FPN_fuse(feature_channels, fpn_out=fpn_out)
        
        self.head = nn.Conv2d(fpn_out, out_channel, kernel_size=3, padding=1)
        
        # convert_relu_to_swish(self)

    def forward(self, x, label=None):
        input_size = (x.size()[2], x.size()[3])

        features = self.backbone(x)
        features[-1] = self.PPN(features[-1])
        x = self.head(self.FPN(features))

        x = F.interpolate(x, size=input_size, mode='bilinear')
        
        return x


class MixPretrainNet(nn.Module):
    # Implementing only the object path
    def __init__(self, num_features, out_channel, backbone='resnet34', pretrained=False, fpn_out=64):
        super(MixPretrainNet, self).__init__()

        self.upernet = UperNet(num_features, out_channel, backbone, pretrained, fpn_out)
        
        num_features = self.upernet.feature_channels[-1]
        
        self.proj_mean = nn.Linear(num_features, 256)
        self.proj_max = nn.Linear(num_features, 256)
        self.proj_min = nn.Linear(num_features, 256)
        
        # convert_relu_to_swish(self)

    def forward(self, x, label=None):
        input_size = (x.size()[2], x.size()[3])

        raw_features = self.upernet.backbone(x)
        
        temp_f = raw_features[-1]

        B = x.shape[0] // 2
        raw_features = [f[:] for f in raw_features]
        raw_features[-1] = self.upernet.PPN(raw_features[-1])
        x = self.upernet.head(self.upernet.FPN(raw_features))
        x = F.interpolate(x, size=input_size, mode='bilinear')
        
        feature = temp_f
        mean_feature = torch.mean(feature, dim=(2, 3))
        max_feature = torch.amax(feature, dim=(2, 3))
        min_feature = torch.amin(feature, dim=(2, 3))
        
        features = (
            self.proj_mean(mean_feature) +
            self.proj_max(max_feature) +
            self.proj_min(min_feature)
        )

        features = features / features.norm(dim=-1, keepdim=True)
        
        return features, x


class PretrainNet(nn.Module):
    # Implementing only the object path
    def __init__(self, num_features, out_channel, backbone='resnet34', pretrained=False, fpn_out=64):
        super(PretrainNet, self).__init__()

        self.upernet = UperNet(num_features, out_channel, backbone, pretrained, fpn_out)
        
        num_features = self.upernet.feature_channels[-1]
        
        self.proj_mean = nn.Linear(num_features, 256)
        self.proj_max = nn.Linear(num_features, 256)
        self.proj_min = nn.Linear(num_features, 256)
        
        # convert_relu_to_swish(self)

    def forward(self, x, label=None):
        input_size = (x.size()[2], x.size()[3])

        feature = self.upernet.backbone(x)[-1]
        mean_feature = torch.mean(feature, dim=(2, 3))
        max_feature = torch.amax(feature, dim=(2, 3))
        min_feature = torch.amin(feature, dim=(2, 3))
        
        features = (
            self.proj_mean(mean_feature) +
            self.proj_max(max_feature) +
            self.proj_min(min_feature)
        )

        features = features / features.norm(dim=-1, keepdim=True)
        
        return features
