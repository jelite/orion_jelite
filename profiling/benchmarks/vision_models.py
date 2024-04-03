import os
from platform import node
import sched
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
from torchvision import models, datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.multiprocessing import Pool, Process, set_start_method, Manager, Value, Lock
from datetime import timedelta
import random
import numpy as np
import time
import os
import argparse
import threading

        
def vision(model_name, batch_size, local_rank, do_eval, is_additional):

    data = torch.randn(batch_size, 3, 224, 224, device=torch.device("cuda"))
    data = torch.ones([batch_size, 3, 224, 224], pin_memory=True).to(local_rank)
    target = torch.ones([batch_size], pin_memory=True).to(torch.long).to(local_rank)
    model = models.__dict__[model_name](num_classes=1000)
    model = model.to(local_rank)
    
    print(f"additional : {is_additional}")
    if do_eval:
        model.eval()
    else:
        model.train()
        optimizer =  torch.optim.SGD(model.parameters(), lr=0.1)
        criterion =  torch.nn.CrossEntropyLoss().to(local_rank)

    #warm up
    if is_additional:
        for batch_idx in range(5): #batch in train_iter:
            if do_eval:
                with torch.no_grad():
                    output = model(data)
            else:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
    torch.cuda.profiler.cudart().cudaProfilerStart()
    if do_eval:
        with torch.no_grad():
            output = model(data)
    else:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    torch.cuda.profiler.cudart().cudaProfilerStop()
