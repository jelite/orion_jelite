import time
from torchvision import models
import torch

data = torch.randn(1, 3, 224, 224, device=torch.device("cuda"))
data = torch.ones([1, 3, 224, 224], pin_memory=True).to(0)
target = torch.ones([1], pin_memory=True).to(torch.long).to(0)


model_names = ["densenet121", "resnet50", "mobilenet_v3_large", "efficientnet_v2_m"]
model_names = ["efficientnet_v2_m"]
for model_name in model_names:
    model = models.__dict__[model_name](num_classes=1000)
    model = model.to(0)

    model.eval()

    times = []
    with torch.no_grad():
        for _ in range(10):
            output = model(data)

        for _ in range(100):
            start = time.time()
            output = model(data)
            torch.cuda.synchronize()
            end = time.time()
            times.append((end-start)*1000)

    print(f"{model_name},{sum(times[90:])/10}!!!")

