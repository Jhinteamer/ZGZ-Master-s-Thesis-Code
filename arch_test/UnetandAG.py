import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torch.nn.parameter import Parameter
import pretrainedmodels
import os
from collections import OrderedDict
from torchstat import stat

def _sanet(arch, block, layers, pretrained, **kwargs):
    model = SA_ResNet(block, layers, **kwargs)

    # 暂时不去读与训练模型
    if pretrained:
        state_dict = load_state_dict('pretrained_model/sa_resnet50/sa_resnet50_210310.pth.tar')
        model.load_state_dict(state_dict, strict=False)
    return model

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.sa_layer = sa_layer(F_l)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)

class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        # 以下两个卷积控制维度
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x

#Unet with Resnet

class sa_layer(nn.Module):
    """Constructs a Channel Spatial Group module.

    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=64):
        super(sa_layer, self).__init__()

        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class SABottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(SABottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride=stride, groups=groups, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.sa = sa_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.sa(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class SA_ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(SA_ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, SABottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
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

        return x

def load_state_dict(checkpoint_path):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'
        if state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        print("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        print("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()

def _sanet(arch, block, layers, pretrained, **kwargs):
    model = SA_ResNet(block, layers, **kwargs)

    # 暂时不去读与训练模型
    if pretrained:
        state_dict = load_state_dict('E:/ZGZ/pytorch-nested-unet-master/pretrained_model/sa_resnet50/sa_resnet50_210310.pth.tar')
        model.load_state_dict(state_dict, strict=False)
    return model

def sa_resnet50(pretrained=False):
    model = _sanet('SANet-50', SABottleneck, [3, 4, 6, 3], pretrained=pretrained)
    return model

def conv3x3BR(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                         nn.BatchNorm2d(out_planes),
                         nn.ReLU())

class UNetplusWithSAResnet50EncoderandAGsSA(nn.Module):
    DEPTH = 6

    def __init__(self, n_classes=2, input_channels=3, deep_supervision = False, input_h = 224, input_w = 160,**kwargs):
        super().__init__()
        resnet = sa_resnet50(pretrained=True)
        down_blocks = []

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)

        self.expand1 = conv3x3BR(3,64)
        self.expand2 = conv3x3BR(3,64)
        self.expand3 = conv3x3BR(3,512)
        self.expand4 = conv3x3BR(3,1024)

        self.shink1 = conv3x3BR(128,64)
        self.shink2 = conv3x3BR(128,64)
        self.shink3 = conv3x3BR(1024,512)
        self.shink4 = conv3x3BR(2048,1024)


        # TODO

        # 2048 up 1024 1024 + 1024 = 2048 up = 1024
        self.Up5 = up_conv(ch_in=2048, ch_out=1024)
        self.Att5 = Attention_block(F_g=1024, F_l=1024, F_int=512)
        self.conv3x3_5 = conv3x3(1024,1024)
        self.sa_layer5 = sa_layer(channel=1024)
        self.Up_conv5 = conv_block(ch_in=2048, ch_out=1024)

        # 1024 up 512 512 + 512 = 1024 up = 512
        self.Up4 = up_conv(ch_in=1024, ch_out=512)
        self.Att4 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.conv3x3_4 = conv3x3(512, 512)
        self.sa_layer4 = sa_layer(channel=512)
        self.Up_conv4 = conv_block(ch_in=1024, ch_out=512)

        # 512 up 256 256 + 256 = 512 up = 256
        self.Up3 = up_conv(ch_in=512, ch_out=256)
        self.Att3 = Attention_block(F_g=256, F_l=256, F_int=64)
        self.conv3x3_3 = conv3x3(256, 256)
        self.sa_layer3 = sa_layer(channel=256)
        self.Up_conv3 = conv_block(ch_in=512, ch_out=256)

        # 256 up 128 128 + 64 = 172 up = 128
        self.Up2 = up_conv(ch_in=256, ch_out=128)
        self.Att2 = Attention_block(F_g=128, F_l=64, F_int=64 // 2)

        self.conv3x3_2 = conv3x3(64, 64)
        self.sa_layer2 = sa_layer(channel=64, groups=32)
        self.Up_conv2 = conv_block(ch_in=128 + 64, ch_out=128)

        # 128 up 64 64 + 3 = 67 up = 64
        self.Up1 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv1 = conv_block(64 + 3, ch_out=64)


        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

    def forward(self, x, with_output_feature_map=False):
        img0 = x
        img1 = self.avg_pool(img0)
        img2 = self.avg_pool(img1)
        img3 = self.avg_pool(img2)
        img4 = self.avg_pool(img3)
        # img = dict()

        pre_pools = dict()
        pre_pools[f"layer_0"] = x   # s=1 c=3
        x = self.input_block(x)
        x = self.shink1(torch.cat([x,self.expand1(img1)],dim=1))
        pre_pools[f"layer_1"] = x   # s=2 c=64
        x = self.input_pool(x)
        x = self.shink2(torch.cat([x, self.expand2(img2)], dim=1))

        x = self.down_blocks[0](x)
        pre_pools[f"layer_2"] = x

        x = self.down_blocks[1](x)
        pre_pools[f"layer_3"] = x
        x = self.shink3(torch.cat([x, self.expand3(img3)], dim=1))

        x = self.down_blocks[2](x)
        pre_pools[f"layer_4"] = x
        x = self.shink4(torch.cat([x, self.expand4(img4)], dim=1))

        x = self.down_blocks[3](x)

        # down_block 和 up_block 中各有五个元素
        # flayer0 s=1 c=3
        # layer0
        # layer1
        # layer2
        # flayer1 s=2 c=64
        # layer3
        # layer4 i=2
        # flayer2 s=4 c=256
        # layer5 i=3
        # (out) flayer3 s=8 c=512
        # layer6 i=4
        # flayer4 s=16 c=1024
        # layer7 i=5 (continue)
        # x s=32 c=2048

        x = self.bridge(x)

        # decoding + concat path

        # 2048 up 1024 1024 + 1024 = 2048 up = 1024
        d5 = self.Up5(x)
        x4 = self.Att5(g=d5, x=pre_pools[f"layer_4"])
        x4 = self.conv3x3_5(x4)
        x4 = self.sa_layer5(x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        # 1024 up 512 512 + 512 = 1024 up = 512
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=pre_pools[f"layer_3"])
        x3 = self.conv3x3_4(x3)
        x3 = self.sa_layer4(x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        # 512 up 256 256 + 256 = 512 up = 256
        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=pre_pools[f"layer_2"])
        x2 = self.conv3x3_3(x2)
        x2 = self.sa_layer3(x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        # 256 up 128 128 + 64 = 172 up = 128
        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=pre_pools[f"layer_1"])
        x1 = self.conv3x3_2(x1)
        x1 = self.sa_layer2(x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        # 128 up 64 64 + 3 = 67 up = 64
        d1 = self.Up1(d2)

        d1 = torch.cat((pre_pools[f"layer_0"], d1), dim=1)
        d1 = self.Up_conv1(d1)

        x = d1

        output_feature_map = x
        x = self.out(x)
        del pre_pools
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x

def main():
    net = UNetplusWithSAResnet50EncoderandAGsSA(num_classes=2)
    #print(net)
    input = torch.rand(1, 3, 160, 224)

    print(net(input).shape)

    stat(net, (3,160,224))



if __name__ == '__main__':
    main()