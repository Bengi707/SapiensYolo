import torch
from lite.demo.extract_feature import load_model

model = load_model(r"C:\Users\Bengi\Downloads\sapiens_0.3b_epoch_1600_clean.pth", True)
model.to(torch.bfloat16)
model.to('cuda:3')
inputs = torch.rand((1,3,1024,1024)).to("cuda:0").to(torch.bfloat16)
outputs = model(inputs)
global_model = list(model.children())[1]
global_model(inputs).shape