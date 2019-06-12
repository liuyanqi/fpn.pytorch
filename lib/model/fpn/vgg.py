import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from model.utils.config import cfg
from model.fpn.fpn import _FPN
import torchvision.models as models

__all__ = [
 "vgg16"
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


# class VGG16(nn.Module):

#     def __init__(self, num_classes=1000, init_weights=True):
#         super(VGG16, self).__init__()
#         self.layer1 = self._make_layers([64, 64]);
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.layer2 = self._make_layers([128, 128])
#         self.layer3 = self._make_layers([256, 256, 256])
#         self.layer4 = self._make_layers([512, 512 ,512])
#         self.layer5 = self._make_layers([512, 512, 512])
#         self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
#         self.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, num_classes),
#         )
#         if init_weights:
#             self._initialize_weights()

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.maxpool(x)
#         x = self.layer2(x)
#         x = self.maxpool(x)
#         x = self.layer3(x)
#         x = self.maxpool(x)
#         x = self.layer4(x)
#         x = self.maxpool(x)
#         x = self.layer5(x)
#         x = self.maxpool(x)

#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)


#     def _make_layers(self, cfg, batch_norm=False):
#         layers = []
#         in_channels = 3
#         for v in cfg:
#             if v == 'M':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             else:
#                 conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#                 if batch_norm:
#                     layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#                 else:
#                     layers += [conv2d, nn.ReLU(inplace=True)]
#                 in_channels = v
#         return nn.Sequential(*layers)


# cfgs = {
#     'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     '''this one'''
#     'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }


# def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
#     if pretrained:
#         kwargs['init_weights'] = False
#     model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls[arch],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model




# def _vgg16(pretrained=False, progress=True, **kwargs):
#     """VGG 16-layer model (configuration "D")

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     model = VGG16()
#     if pretrained:
#         kwargs["init_weights"] = False
#         state_dict = model_zoo.load_url(model_urls['vgg16'], progress=progress)
#         model.load_state_dict(state_dict)

#     return model


class vgg16(_FPN):
    def __init__(self, classes, pretrained=False, class_agnostic=False):
        self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
        self.dout_base_model = 256
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic

        _FPN.__init__(self, classes, class_agnostic)

    def _init_modules(self):
        vgg16 = models.vgg16()

        if self.pretrained ==True:
            state_dict = torch.load(self.model_path)
            vgg16.load_state_dict({k:v for k,v in state_dict.items() if k in vgg16.state_dict()})

        # not using the last maxpool layer
        self.RCNN_base = nn.Sequential(*list(vgg16.features._modules.values())[:-1])
      # (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      # (1): ReLU(inplace)
      # (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      # (3): ReLU(inplace)
      # (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      # (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      # (6): ReLU(inplace)
      # (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      # (8): ReLU(inplace)
      # (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      # (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      # (11): ReLU(inplace)
      # (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      # (13): ReLU(inplace)
      # (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      # (15): ReLU(inplace)
      # (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      # (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      # (18): ReLU(inplace)
      # (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      # (20): ReLU(inplace)
      # (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      # (22): ReLU(inplace)
      # (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      # (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      # (25): ReLU(inplace)
      # (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      # (27): ReLU(inplace)
      # (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      # (29): ReLU(inplace)

        # print(self.RCNN_base)

        self.RCNN_layer0 = nn.Sequential(self.RCNN_base[:5])
        # print(self.RCNN_layer0)
        self.RCNN_layer1 = nn.Sequential(self.RCNN_base[5:10])
        # print(self.RCNN_layer1)
        self.RCNN_layer2 = nn.Sequential(self.RCNN_base[10:17])
        # print(self.RCNN_layer2)
        self.RCNN_layer3 = nn.Sequential(self.RCNN_base[17:24])
        # print(self.RCNN_layer3)
        self.RCNN_layer4 = nn.Sequential(self.RCNN_base[24:30])
        # print(self.RCNN_layer4)

        # Fix the layers before conv3
        for p in self.RCNN_layer0.parameters(): p.requires_grad=False
        for p in self.RCNN_layer1.parameters(): p.requires_grad=False

        self.RCNN_toplayer = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0) #reduce channel

        # Smooth layers
        self.RCNN_smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.RCNN_smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.RCNN_smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        #Lateral layers
        self.RCNN_latlayer1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.RCNN_latlayer2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.RCNN_latlayer3 = nn.Conv2d( 128, 256, kernel_size=1, stride=1, padding=0)

        self.RCNN_roi_feat_ds = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        self.RCNN_top = nn.Sequential(
          nn.Conv2d(256, 1024, kernel_size=cfg.POOLING_SIZE, stride=cfg.POOLING_SIZE, padding=0),
          nn.ReLU(True),
          nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
          nn.ReLU(True)
          )

        self.RCNN_cls_score = nn.Linear(1024, self.n_classes)
        if(self.class_agnostic):
            self.RCNN_bbox_pred = nn.Linear(1024, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(1024, 4 * self.n_classes)



    def train(self, mode=True):
        nn.Module.train(self, mode)
        if mode:
            self.RCNN_layer0.eval()
            self.RCNN_layer1.eval()
            self.RCNN_layer2.train()
            self.RCNN_layer3.train()
            self.RCNN_layer4.train()

            self.RCNN_smooth1.train()
            self.RCNN_smooth2.train()
            self.RCNN_smooth3.train()

            self.RCNN_latlayer1.train()
            self.RCNN_latlayer2.train()
            self.RCNN_latlayer3.train()

            self.RCNN_toplayer.train()

    def _head_to_tail(self, pool5):
        # print("pool5", pool5.size())
        fc7 = self.RCNN_top(pool5)
        fc7 = fc7.mean(3).mean(2)
        # print(fc7.size())
        return fc7

