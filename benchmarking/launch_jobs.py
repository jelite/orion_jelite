import argparse
import json
import threading
import time
from ctypes import *
import os
import sys
import numpy as np

import json


from torchvision import models
import torch
import torch.multiprocessing as mp

home_directory = os.path.expanduser( '~' )
sys.path.append(f"{home_directory}/DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL/pytorch")
sys.path.append(f"{home_directory}/DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL/pytorch/utils")
from benchmark_suite.transformer_trainer import transformer_loop
sys.path.append(f"{home_directory}/DeepLearningExamples/PyTorch/LanguageModeling/BERT")
from bert_trainer import bert_loop

from benchmark_suite.train_imagenet import imagenet_loop
from benchmark_suite.toy_models.bnorm_trainer import bnorm_loop
from benchmark_suite.toy_models.conv_bn_trainer import conv_bn_loop

from src.scheduler_frontend import PyScheduler

function_dict = {
    "resnet50": imagenet_loop,
    "resnet101": imagenet_loop,
    "mobilenet_v2": imagenet_loop,
    "mobilenet_v3_large": imagenet_loop,
    "bnorm": bnorm_loop,
    "conv_bnorm": conv_bn_loop,
    "bert": bert_loop,
    "transformer": transformer_loop,
    "efficientnet_v2_m":imagenet_loop,
    "densenet121":imagenet_loop,
    "vit_l_16":imagenet_loop,
    "swin_b":imagenet_loop
}


class Request():
    def __init__(self, id, start_time) -> None:
        self.id = id
        self.start_time = start_time
        self.end_time = 0
        
    def get_duration(self):
        return f"{(self.end_time - self.start_time)*1000}"
    
def seed_everything(seed: int):
    import random, os
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def launch_jobs(config_dict_list, input_args, run_eval):
    seed_everything(42)

    num_clients = len(config_dict_list)
    s = torch.cuda.Stream()

    # init
    num_barriers = num_clients+1
    barriers = [threading.Barrier(num_barriers) for i in range(num_clients)]
    client_barrier = threading.Barrier(num_clients)
    home_directory = os.path.expanduser( '~' )
    if run_eval:
        sched_lib = cdll.LoadLibrary(home_directory + "/orion/src/scheduler/scheduler_eval.so")
    else:
        
        sched_lib = cdll.LoadLibrary(home_directory + "/orion/src/scheduler/scheduler.so")
    py_scheduler = PyScheduler(sched_lib, num_clients)

    model_names = [config_dict['arch'] for config_dict in config_dict_list]
    model_files = [config_dict['kernel_file'] for config_dict in config_dict_list]

    additional_model_files = [config_dict['additional_kernel_file'] if 'additional_kernel_file' in config_dict else None for config_dict in config_dict_list]
    num_kernels = [config_dict['num_kernels'] for config_dict in config_dict_list]
    num_iters = [config_dict['num_iters'] for config_dict in config_dict_list]
    train_list = [config_dict['args']['train'] for config_dict in config_dict_list]
    additional_num_kernels = [config_dict['additional_num_kernels'] if 'additional_num_kernels' in config_dict else None  for config_dict in config_dict_list]
    tids = []
    threads = []
    config_train = config_dict_list[0]
    config_infer = config_dict_list[1]
    rps = config_infer['args']['rps']
    latency_bound = config_infer['args']['latency_bound']
    rps_start_barrier = mp.Barrier(2)
    warmup_event = mp.Event()
    end_event = mp.Event()
    request_queue = mp.Queue()
    
    if rps > 0:
        if config_infer['args']['uniform']:
            sleep_times = [1/rps]*(num_iters[1]-200)
        else:
            sleep_times = np.random.exponential(scale=1/rps, size=num_iters[1])
            
    #train
    file_name = f'{config_train["arch"]}x{config_infer["arch"]}_{rps}_{latency_bound}ms'
    
    
    func = function_dict[config_train['arch']]
    model_args = config_train['args']
    model_args.update({"num_iters":num_iters[0], "local_rank": 0, "barriers": barriers, \
                    "client_barrier": client_barrier, "tid": 0, \
                    "file_name":file_name, "warmup_event":warmup_event, "do_save":input_args.do_save, "trial":input_args.trial,
                    "end_event":end_event
                    })
    thread = threading.Thread(target=func, kwargs=model_args)
    thread.start()
    tids.append(thread.native_id)
    threads.append(thread)
    
    #infer
    func = function_dict[config_infer['arch']]
    model_args = config_infer['args']
    model_args.update({"num_iters":num_iters[1], "local_rank": 0, "barriers": barriers, \
                    "client_barrier": client_barrier, "tid": 1, \
                    "rps_start_barrier":rps_start_barrier, "request_queue": request_queue,"file_name":file_name, "warmup_event":warmup_event, "do_save":input_args.do_save, "trial":input_args.trial,
                    "end_event":end_event
                    })
    thread = threading.Thread(target=func, kwargs=model_args)
    thread.start()
    tids.append(thread.native_id)
    threads.append(thread)
    
    sched_thread = threading.Thread(
        target=py_scheduler.run_scheduler,
        args=(
            barriers,
            tids,
            model_names,
            model_files,
            additional_model_files,
            num_kernels,
            additional_num_kernels,
            num_iters,
            True,
            run_eval,
            input_args.algo=='reef',
            input_args.algo=='sequential',
            input_args.reef_depth if input_args.algo=='reef' else input_args.orion_max_be_duration,
            input_args.orion_hp_limit,
            input_args.orion_start_update,
            train_list,
        )
    )
    print("sche start")
    sched_thread.start()



    path = f"./overall_test/arrival_times-rps{rps}-reqs{num_iters[1]-200}-num{input_args.trial}.json"


    with open(path, "r") as json_file:
        json_data = json.load(json_file)




    rps_start_barrier.wait()
    print(f"RPS{rps} READY!!!")
    for idx, data in enumerate(json_data):
        st_time = time.time()
        req = Request(idx, st_time)
        request_queue.put(req)
        time.sleep(float(data))

    # for id in range(num_iters[1]-200):
    #     st_time = time.time()
    #     req = Request(id, st_time)
    #     request_queue.put(req)
    #     time.sleep(sleep_times[id])
        
    for thread in threads:
        thread.join()

    print("train joined!")
    os.system('kill %d' % os.getpid())
    # sched_thread.join(
    print("sched joined!")

    print("--------- all threads joined!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, required=True,
                        help='choose one of orion | reef | sequential')
    parser.add_argument('--config_file', type=str, required=True,
                        help='path to the experiment configuration file')
    parser.add_argument('--reef_depth', type=int, default=1,
                        help='If reef is used, this stands for the queue depth')
    parser.add_argument('--orion_max_be_duration', type=int, default=1,
                        help='If orion is used, the maximum aggregate duration of on-the-fly best-effort kernels')
    parser.add_argument('--orion_start_update', type=int, default=1,
                        help='If orion is used, and the high priority job is training, this is the kernel id after which the update phase starts')
    parser.add_argument('--orion_hp_limit', type=int, default=1,
                        help='If orion is used, and the high priority job is training, this shows the maximum tolerated training iteration time')
    parser.add_argument('--do_save', action='store_true')
    parser.add_argument('--trial', type=int)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    with open(args.config_file) as f:
        config_dict = json.load(f)
    launch_jobs(config_dict, args, True)
