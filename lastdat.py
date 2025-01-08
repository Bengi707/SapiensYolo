import torch
from matplotlib.colors import to_rgb

from pretrain.mmpretrain.models import VisionTransformer as Sapiens
import torch.nn as nn
from torchvision.transforms import Compose, Normalize
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
        # self.vit = Sapiens()
        # self.vit = Sapiens("0.3b",img_size = 1024, embed_dim = 1280, num_layers=32, num_heads=16, feedforward_channels= 1280*4)
        self.vit = Sapiens("0.3b",img_size = 1024)
        msg = self.vit.load_state_dict(torch.load("sapiens_0.3b_epoch_1600_clean.pth",map_location=torch.device('cuda')))
        # msg = self.vit.load_state_dict(torch.load(r"C:\Users\Bengi\Downloads\sapiens_0.3b_epoch_1600_clean.pth"))
        print(msg)
        for param in self.vit.parameters():
            param.requires_grad = False

        self.normalize = T.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
        self.normalize = self.normalize.to(torch.device('cuda'))

        # P3: Small scale
        self.upsample_p3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample_p3 = self.upsample_p3.to(torch.device('cuda'))

        self.conv_p3 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)
        self.conv_p3 = self.conv_p3.to(torch.device('cuda'))
        # P4: Medium scale
        self.conv_p4 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.conv_p4 = self.conv_p4.to(torch.device('cuda'))

        # P5: Large scale
        self.conv_p5 = nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1)
        self.conv_p5 = self.conv_p5.to(torch.device('cuda'))


    def forward(self, x):
        # print(f"Backbone output: {x.shape}")
        x = x.to("cuda")
        self.vit = self.vit.to("cuda")
        x = self.normalize(x)
        x = self.vit(x)[0]

        # P3
        p3 = self.upsample_p3(x)  # Upsample to [B, 768, 64, 64]
        p3 = self.conv_p3(p3)  # [B, 256, 32, 32]
        # print(f"P3: {p3.shape}")

        # P4
        p4 = self.conv_p4(x)  # [B, 512, 16, 16]
        # print(f"P4: {p4.shape}")

        # P5
        p5 = self.conv_p5(x)  # [B, 1024, 8, 8]
        # print(f"P5: {p5.shape}")

        return p3, p4, p5



# # Create a modified Sapiens ViT model
# modified_sapiens_vit = ModifiedSapiensViT()
#
# # Test with a random input image
# input_image = torch.randn(1, 3, 512, 512)  # Batch size = 1, 3 channels (RGB), 224x224 resolution
# output = modified_sapiens_vit(input_image)
#
# # Print final output shape
# print("Final Output Shape:", output.shape)