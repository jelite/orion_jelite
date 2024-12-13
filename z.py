import time
from torchvision import models
import torch

data = torch.randn(8, 3, 224, 224, device=torch.device("cuda"))

model_names = ["mnasnet1_3", "shufflenet_v2_x2_0", "squeezenet1_1", "mobilenet_v3_large"]

for model_name in model_names:
    print("aa")
    model = models.__dict__[model_name](num_classes=1000)
    model = model.cuda()
    
    
    for _ in range(3):
        model(data)

    # g = torch.cuda.CUDAGraph()
    
    
    # with torch.cuda.graph(g):
    #     static_output = model(data)
    
    # print("start")
    # torch.cuda.synchronize()
    # durs = []




    # for _ in range(100):
    #     start = time.time()
    #     g.replay()
    #     end = time.time()
    #     durs.append((end-start)*1000)
    # print(f"{model_name},{sum(durs[90:])/10}!!!")
