import torch
from torch import nn
import torch.nn.functional as F

import models.resnet

# DeepLabV3+.
class DeepLabv3plus_backbone(nn.Module):
    def __init__(self, pretrained, layers):
        super(DeepLabv3plus_backbone, self).__init__()

        # Encoder part
        assert(layers in [50, 101, 152])
        if layers == 50:
            resnet = models.resnet.resnet50(pretrained=pretrained)
            resnet_thermal = models.resnet.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet.resnet101(pretrained=pretrained)
            resnet_thermal = models.resnet.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet.resnet152(pretrained=pretrained)
            resnet_thermal = models.resnet.resnet152(pretrained=pretrained)
        # RGB stream
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu,
                                    resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        # Thermal stream
        self.layer0_thermal = nn.Sequential(resnet_thermal.conv1, resnet_thermal.bn1, resnet_thermal.relu,
                                    resnet_thermal.conv2, resnet_thermal.bn2, resnet_thermal.relu,
                                    resnet_thermal.conv3, resnet_thermal.bn3, resnet_thermal.relu,
                                    resnet_thermal.maxpool)
        self.layer1_thermal, self.layer2_thermal, self.layer3_thermal, self.layer4_thermal = \
            resnet_thermal.layer1, resnet_thermal.layer2, resnet_thermal.layer3, resnet_thermal.layer4
        for n, m in self.layer3_thermal.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4_thermal.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)


        # ASPP part FOR both RGB and thermal streams
        in_dim = 2048
        reduction_dim = 256
        self.features = []
        self.features_thermal = []
        Aspp_rates = [1, 6, 12, 18]
        for rate in Aspp_rates:
            if rate == 1:
                kernel_size = 1
                padding = 0
            else:
                kernel_size = 3
                padding = rate

            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=kernel_size, stride=1, padding=padding, dilation=rate,
                          bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU()
            ))
            self.features_thermal.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=kernel_size, stride=1, padding=padding, dilation=rate,
                          bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU()
            ))
        self.features.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(in_dim, reduction_dim, 1, stride=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU()
                                           ))
        self.features_thermal.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                           nn.Conv2d(in_dim, reduction_dim, 1, stride=1, bias=False),
                                           nn.BatchNorm2d(reduction_dim),
                                           nn.ReLU()
                                           ))

        self.features = nn.ModuleList(self.features)
        self.features_thermal = nn.ModuleList(self.features_thermal)

        self.conv_layer = nn.Sequential(
            nn.Conv2d(reduction_dim * 5, reduction_dim, 1, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.conv_layer_thermal = nn.Sequential(
            nn.Conv2d(reduction_dim * 5, reduction_dim, 1, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.ReLU(),
            nn.Dropout(0.5))


        # Decoder Part
        self.low_conv = nn.Sequential(
            nn.Conv2d(256, reduction_dim, 1, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.ReLU()
        )
        self.last_conv = nn.Sequential(
            nn.Conv2d(reduction_dim * 2, reduction_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(reduction_dim, reduction_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.ReLU())

        self.low_conv_thermal = nn.Sequential(
            nn.Conv2d(256, reduction_dim, 1, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.ReLU()
        )
        self.last_conv_thermal = nn.Sequential(
            nn.Conv2d(reduction_dim * 2, reduction_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(reduction_dim, reduction_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.ReLU())

        self.fused_conv = nn.Sequential(
            nn.Conv2d(reduction_dim * 2, reduction_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(reduction_dim, reduction_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.ReLU())

        self._init_weight()



    def forward(self, x, thermal):

        '''RGB Input Stream'''
        # Encoder Part
        x = self.layer0(x)
        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)

        # ASPP part
        x_size = x.size()
        out = []
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        x = self.conv_layer(torch.cat(out, 1))

        # Decoder Part
        low_level_feat = self.low_conv(low_level_feat)
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        '''Thermal Input Stream'''
        # Encoder Part
        tt = self.layer0_thermal(thermal)
        tt = self.layer1_thermal(tt)
        low_level_feat_thermal = tt
        tt = self.layer2_thermal(tt)
        tt_tmp = self.layer3_thermal(tt)
        tt = self.layer4_thermal(tt_tmp)

        # ASPP part
        tt_size = tt.size()
        out_tt = []
        for f in self.features_thermal:
            out_tt.append(F.interpolate(f(tt), tt_size[2:], mode='bilinear', align_corners=True))
        tt = self.conv_layer_thermal(torch.cat(out_tt, 1))

        # Decoder Part
        low_level_feat_thermal = self.low_conv(low_level_feat_thermal)
        tt = F.interpolate(tt, size=low_level_feat_thermal.size()[2:], mode='bilinear', align_corners=True)
        tt = torch.cat((tt, low_level_feat_thermal), dim=1)
        tt = self.last_conv_thermal(tt)

        '''Fusion Stream'''
        fused_fea = torch.cat((x, tt), dim=1)
        fused_fea = self.fused_conv(fused_fea)

        return x, tt, fused_fea

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # m.weight.data.zero_()
                nn.init.normal_(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabPredict(nn.Module):
    def __init__(self, classes):
        super(DeepLabPredict, self).__init__()


        self.cls_rgb = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(256, classes, kernel_size=1)
        )
        self.cls_thermal = nn.Sequential(
            nn.Conv2d(256 * 2 + classes, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(256, classes, kernel_size=1)
        )
        self.cls_fusion = nn.Sequential(
            nn.Conv2d(256 * 3 + classes * 2, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(256, classes, kernel_size=1)
        )
        self.cls = nn.Sequential(
            nn.Conv2d(classes * 3, classes, kernel_size=1)
        )

        self._init_weight()



    def forward(self, f_x, f_t, f_f, h, w):

        aux_RGB = self.cls_rgb(f_x)

        f_t = torch.cat((f_x, aux_RGB, f_t), dim=1)
        aux_thermal = self.cls_thermal(f_t)

        f_f = torch.cat((f_t, aux_thermal, f_f), dim=1)
        aux_fusion = self.cls_fusion(f_f)

        final = self.cls(torch.cat((aux_RGB, aux_thermal, aux_fusion),dim=1))

        aux_RGB = F.interpolate(aux_RGB, size=(h, w), mode='bilinear', align_corners=True)
        aux_thermal = F.interpolate(aux_thermal, size=(h, w), mode='bilinear', align_corners=True)
        aux_fusion = F.interpolate(aux_fusion, size=(h, w), mode='bilinear', align_corners=True)
        final = F.interpolate(final, size=(h, w), mode='bilinear', align_corners=True)

        return final, aux_RGB, aux_thermal, aux_fusion


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # m.weight.data.zero_()
                nn.init.normal_(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == '__main__':
    from thop import profile
    model = DeepLabv3plus_backbone(pretrained=False, layers=50)

    image = torch.randn(1, 3, 320, 480)
    thermal = torch.randn(1, 3, 320, 480)

    flops, params = profile(model, inputs=(image, thermal))


    print('Params: %.1f (M)' % (params / 1000000.0))
    print('FLOPS: %.1f (G)' % (flops / 1000000000.0))