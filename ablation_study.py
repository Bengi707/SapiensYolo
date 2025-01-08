import torch

from pretrain.mmpretrain.models import VisionTransformer as Sapiens
import torch.nn as nn
import torchvision.transforms as T


class Dummy(nn.Module):
    def __init__(self,dim, index, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = index
        self.dim = dim
    def forward(self,x):
        return x[self.index]

class ModifiedSapiensViT(nn.Module):
    def __init__(self):
        super(ModifiedSapiensViT, self).__init__()
        self.vit = Sapiens("0.3b", img_size=1024)
        msg = self.vit.load_state_dict(torch.load(r"C:\Users\Bengi\Downloads\sapiens_0.3b_epoch_1600_clean.pth",map_location=torch.device('cuda')))
        print(msg)
        for param in self.vit.parameters():
            param.requires_grad = False

        self.normalize = T.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
        self.normalize = self.normalize.to(torch.device('cuda'))

        # P3: Small scale
        self.upsample_p3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.conv_p3 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)

        # P4: Medium scale
        self.upsample_p4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_p4 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)

        # P5: Large scale
        self.conv_p5 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.normalize(x)
        x = self.vit(x)[0]

        # P3
        p3 = self.upsample_p3(x)
        p3 = self.conv_p3(p3)

        # P4
        p4 = self.upsample_p4(x)
        p4 = self.conv_p4(p4)

        # P5
        p5 = self.conv_p5(x)

        return p3, p4, p5

