import torch
from pretrain.mmpretrain.models import VisionTransformer

# Define dummy input
batch_size = 3
channels = 3  # RGB image
height, width = 224, 224  # Typical input size for VisionTransformer

dummy_input = torch.randn(batch_size, channels, height, width)

# Initialize VisionTransformer
# vit = VisionTransformer(arch="0.3b")  # Adjust params if needed
vit = VisionTransformer()  # Adjust params if needed

# Forward pass
try:
    output = vit(dummy_input)
    print("VisionTransformer output shape:", output[0].shape)
except Exception as e:
    print("Error during forward pass:", e)
