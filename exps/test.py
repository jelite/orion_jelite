import torch
import torchvision

a = torch.rand([1,3,224,224]).cuda()
m = torchvision.models.resnet50().cuda()

m(a)
print("Done")
