import os
from platform import node
import sched
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
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
import json
from ctypes import *


        
def seed_everything(seed: int):
    import random, os
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class DummyDataLoader():
    def __init__(self, batchsize):
        self.batchsize = batchsize
        self.data = torch.rand([self.batchsize, 3, 224, 224], pin_memory=True)
        self.target = torch.ones([self.batchsize], pin_memory=True, dtype=torch.long)

    def __iter__(self):
        return self

    def __next__(self):
        return self.data, self.target



def block(backend_lib, it):
    # block client until request served
    backend_lib.block(it)

def check_stop(backend_lib):
    return backend_lib.stop()

def imagenet_loop(
    model_name,
    batchsize,
    train,
    num_iters,
    rps,
    latency_bound,
    uniform,
    dummy_data,
    local_rank,
    barriers,
    client_barrier,
    tid,
    warmup_event,
    do_save,
    trial,
    end_event,
    input_file='',
    rps_start_barrier=None,
    request_queue=None,
    file_name=None
):

    seed_everything(time.time_ns() % 10**9)
    
    backend_lib = cdll.LoadLibrary(os.path.expanduser('~') + "/orion/src/cuda_capture/libinttemp.so")

    barriers[0].wait()
    print("-------------- thread id:  ", threading.get_native_id())

    if ((train=="train" or train=="be_infer") and tid==1):
        time.sleep(5)

    model = models.__dict__[model_name](num_classes=1000)
    model = model.to(0)

    if train=="train":
        model.train()
        # optimizer =  torch.optim.Adam(model.parameters(), lr=0.1)
        optimizer =  torch.optim.SGD(model.parameters(), lr=0.1)
        criterion =  torch.nn.CrossEntropyLoss().to(local_rank)
    else:
        model.eval()

    if dummy_data:
        train_loader = DummyDataLoader(batchsize)

    train_iter = enumerate(train_loader)
    batch_idx, batch = next(train_iter)
    gpu_data, gpu_target = batch[0].to(local_rank), batch[1].to(local_rank)
    print("Enter loop!")

    #  open loop
    next_startup = time.time()
    open_loop = True
    warmup_flag = True
    
    if train=="be_infer":
        if rps > 0:
            if uniform:
                sleep_times = [1/rps]*(num_iters)
            else:
                sleep_times = np.random.exponential(scale=1/rps, size=num_iters)
            
    if True:
        timings=[]
        for i in range(1):
            print("Start epoch: ", i)

            while batch_idx < num_iters:
                start_iter = time.time()

                if train=="train":
                    gpu_data, gpu_target = batch[0].to(local_rank), batch[1].to(local_rank)
            
                    if warmup_event.is_set():
                        start_time = time.time()
                        optimizer.zero_grad()
                        output = model(gpu_data)
                        loss = criterion(output, gpu_target)
                        loss.backward()
                        optimizer.step()
                        block(backend_lib, batch_idx)
                        torch.cuda.synchronize()
                        end_time = time.time()
                        batch_idx, batch = next(train_iter)
                        if do_save:
                            if not os.path.exists("/workspace/exps/train/"):
                                os.system(f"mkdir /workspace/exps/train/")
                            with open(f"/workspace/exps/train/{file_name}.txt",'a') as f:
                                f.write(f"{(end_time-start_time)*1000}, {trial}\n")
                    else:
                        optimizer.zero_grad()
                        output = model(gpu_data)
                        loss = criterion(output, gpu_target)
                        loss.backward()
                        optimizer.step()
                        block(backend_lib, batch_idx)
                        batch_idx, batch = next(train_iter)
                        if (batch_idx == 1): # for backward
                            barriers[0].wait()
                        if batch_idx == 10: # for warmup
                            barriers[0].wait()
                        
                        
                    if check_stop(backend_lib):
                        print("----Train STOP! by orion")
                        break
                    if end_event.is_set():
                        print("----Train STOP! by event")
                        break
                elif train=="be_infer":
                    gpu_data, gpu_target = batch[0].to(local_rank), batch[1].to(local_rank)
            
                    if warmup_event.is_set():
                        print("BBBBBBBB")
                        time.sleep(sleep_times[batch_idx])
                        start_time = time.time()
                        torch.cuda.nvtx.range_push(f"be_infer_{batch_idx}")
                        output = model(gpu_data)
                        torch.cuda.nvtx.range_pop()
                        block(backend_lib, batch_idx)
                        torch.cuda.synchronize()
                        end_time = time.time()
                        batch_idx, batch = next(train_iter)
                        if do_save:
                            if not os.path.exists("/workspace/exps/be_infer/"):
                                os.system(f"mkdir /workspace/exps/be_infer/")
                            with open(f"/workspace/exps/be_infer/{file_name}.txt",'a') as f:
                                f.write(f"{(end_time-start_time)*1000}, {trial}\n")
                    else:
                        print("AAAAAAAAAAAAAAAAAAAAAAAAAA")
                        torch.cuda.nvtx.range_push(f"be_infer_{batch_idx}")
                        output = model(gpu_data)
                        torch.cuda.nvtx.range_pop()
                        block(backend_lib, batch_idx)
                        batch_idx, batch = next(train_iter)
                        if (batch_idx == 1): # for backward
                            barriers[0].wait()
                        if batch_idx == 10: # for warmup
                            barriers[0].wait()
                        
                    if check_stop(backend_lib):
                        print("----be_infer STOP! by orion")
                        break
                    if end_event.is_set():
                        print("----be_infer STOP! by event")
                        break
                    
                else: #infer
                    with torch.no_grad():
                        if not (batch_idx-200) % 500 :
                            print(batch_idx - 200)
                        if batch_idx == 200:
                            warmup_flag = False
                            request_start = time.time()
                        if warmup_flag: 
                            output = model(gpu_data)
                            
                            block(backend_lib, batch_idx)
                            batch_idx,batch = next(train_iter)
                            
                            if (batch_idx == 1 or (batch_idx == 10)):
                                barriers[0].wait()
                        else:
                            queuing_delay = 0
                            is_passed = False
                            if batch_idx == 200:
                                rps_start_barrier.wait()
                                warmup_event.set()
                            req_ = request_queue.get()
                            start_time = time.time()
                            if batch_idx != 200:
                                queuing_delay = (start_time - req_.start_time)*1000
                                
                            if queuing_delay < latency_bound:
                                output = model(gpu_data)
                                block(backend_lib, batch_idx)
                            else:
                                is_passed = True
                                block(backend_lib, -1)
                            torch.cuda.synchronize()        
                            end_time = time.time()
                            batch_idx,batch = next(train_iter)
                            if (batch_idx == 1 or (batch_idx == 10)):
                                barriers[0].wait()
                            if(batch_idx == num_iters-1):
                                request_end = time.time()
                                if do_save:
                                    with open(f"/workspace/exps/{file_name}_total.log",'a') as f:
                                        wall_clock_time = request_end - request_start
                                        f.write(f"{wall_clock_time}, {trial}\n")
                                    
                            req_.end_time = end_time
                            if do_save:
                                with open(f"/workspace/exps/{file_name}.txt",'a') as f:
                                    if is_passed:
                                        f.write(f"passed, {queuing_delay}, {trial}\n")
                                    else:
                                        f.write(f"{req_.get_duration()},{(end_time-start_time)*1000}, {queuing_delay}, {trial}\n")
                            
                                    
                            if check_stop(backend_lib):
                                print("----Infer STOP! by orion")
                                break
                            if end_event.is_set():
                                print("----Infer STOP! by event")
                                break
                            
            if not end_event.is_set():
                print(f"end_event set by {tid}")
                end_event.set()
            

        barriers[0].wait()
        print("Finished! Ready to join!")