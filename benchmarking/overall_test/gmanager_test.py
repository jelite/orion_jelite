import torch

from gmanager import GPUManager

import torch.multiprocessing as mp
import numpy as np

from argparse import ArgumentParser
import os
import time
import json
import copy
import csv
import threading

np.random.seed(time.time_ns() % 1000000000)

do_drop = int(os.getenv("GMANAGER_DROP", 0))
pause = int(os.getenv("GMANAGER_PAUSE", 0))

result_dir = os.path.join(os.path.expanduser("~"), ".results")
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

train_req = {
    "type": "BE",
    "model": "mobilenet_v2",
    "bs": 64,
    "epochs": 1000000,
    "loss_func": "CrossEntropyLoss",
    "optimizer": "SGD",
}
infer_req = {
    "type": "HP",
    "model": "resnet50",
    "warmup": False,
    "bs": 8,
}


def run_gpu_manager(gpu_manager):
    gpu_manager.run(test=True)


if __name__ == "__main__":
    mp.set_start_method("spawn")

    argparser = ArgumentParser()
    argparser.add_argument("--host", type=str, default="127.0.0.1")
    argparser.add_argument("--port", type=int, default=7001)
    argparser.add_argument("--gpu", type=int, default=0)
    argparser.add_argument("--train", type=str, default="resnet50")
    argparser.add_argument("--train-bs", type=int, default=64)
    argparser.add_argument("--infer", type=str, default="resnet50")
    argparser.add_argument("--infer-bs", type=int, default="resnet50")
    argparser.add_argument("--dist", type=str, default="uniform")
    argparser.add_argument("--rps", type=float, default=15)
    argparser.add_argument("--exp-num", type=int, default=0)
    args = argparser.parse_args()

    train_req["model"] = args.train
    if "vit" in args.train or "swin" in args.train:
        if args.train_bs == 32:
            train_req["bs"] = 4
        else:
            train_req["bs"] = 8
    else:
        train_req["bs"] = args.train_bs

    infer_req["model"] = args.infer
    if "vit" in args.infer or "swin" in args.infer:
        infer_req["bs"] = 1
    else:
        infer_req["bs"] = args.infer_bs

    do_tensorrt = int(os.getenv("GMANAGER_TENSORRT", 0))
    do_cuda_graph = int(os.getenv("GMANAGER_CUDA_GRAPH", 0))
    print("============= GPU Manager =============")
    print(f"  Address: {args.host}:{args.port}")
    print(f"  GPU number: {args.gpu}")
    print(f"  TensorRT: {True if do_tensorrt else False}")
    print(f"  CUDA Graph: {True if do_cuda_graph else False}")
    print()
    print(f"  Request Distribution: {args.dist}")
    print(f"  Request per second: {args.rps}")
    print(f"  Training model: {train_req['model']}")
    print(f"  Training batch size: {train_req['bs']}")
    print(f"  Inference model: {infer_req['model']}")
    print(f"  Inference batch size: {infer_req['bs']}")
    print("=======================================")

    gpu_manager = GPUManager(args.host, args.port, args.gpu)

    gpu_manager_runner = threading.Thread(target=run_gpu_manager, args=(gpu_manager,))
    gpu_manager_runner.start()

    time.sleep(8)

    be_req_q = gpu_manager.be_req_q
    hp_req_q = gpu_manager.hp_req_q
    resp_q = gpu_manager.resp_q

    # For warmup
    warmup_infer_req = copy.deepcopy(infer_req)
    warmup_infer_req["warmup"] = True
    warmup_infer_req["enqueue_time"] = time.time() * 1000.
    warmup_infer_req["be_info"] = None
    hp_req_q.put(warmup_infer_req)
    resp = resp_q.get()

    # For profiling
    infer_req["enqueue_time"] = time.time() * 1000.
    infer_req["be_info"] = train_req
    hp_req_q.put(infer_req)
    resp = resp_q.get()

    latency_bound = gpu_manager.latency_bound
    exp_dirname = f"bound{latency_bound:>.4f}-{train_req['model']}-b{train_req['bs']}-{train_req['loss_func']}-{train_req['optimizer']}X{infer_req['model']}-b{infer_req['bs']}-{args.dist}-rps{args.rps}"
    if do_drop and pause:
        result_dir = os.path.join(os.path.expanduser("~"), ".results", f"ours-train-b{args.train_bs}Xinfer-b{args.infer_bs}-{args.exp_num}")
        exp_dir = os.path.join(result_dir, exp_dirname)
    else:
        result_dir = os.path.join(os.path.expanduser("~"), ".results", f"mps-train-b{args.train_bs}Xinfer-b{args.infer_bs}-{args.exp_num}")
        exp_dir = os.path.join(result_dir, exp_dirname)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)

    train_result_file = f"train-result.csv"
    train_result_f = open(os.path.join(exp_dir, train_result_file), 'w')
    train_result_writer = csv.writer(train_result_f)
    train_result_writer.writerow(["req no.", "duration (ms)"])

    infer_result_file = f"infer-result.csv"
    infer_result_f = open(os.path.join(exp_dir, infer_result_file), 'w')
    infer_result_writer = csv.writer(infer_result_f)
    infer_result_writer.writerow(["req no.", "est start time (ms)", "start time (ms)", "end time (ms)", "queuing delay (ms)", "latency (ms)", "status"])

    num_train_reqs = 1
    num_infer_reqs = 2000

    for _ in range(num_train_reqs):
        gpu_manager.running_be_info = train_req
        be_req_q.put(train_req)
    resp = resp_q.get()
    if resp["type"] != "BE" or resp["status"] != "start":
        quit()

    dist_type = args.dist
    rps = args.rps
    if dist_type == "uniform":
        sleep_times = [1/rps] * num_infer_reqs
    elif dist_type == "poisson":
        arrival_times_fname = f'/workspace/gmanager/test/overall_test/arrival_intervals/arrival_intervals-rps{rps}-reqs{num_infer_reqs}-num{args.exp_num}.json'
        with open(arrival_times_fname, "r") as f:
            sleep_times = json.load(f)
    
    start_time = None
    for req in range(num_infer_reqs):
        if req > 0:
            time.sleep(sleep_times[req-1])
        infer_req["enqueue_time"] = time.time() * 1000.
        infer_req["be_info"] = gpu_manager.running_be_info
        hp_req_q.put(infer_req)

    train_num = 0
    infer_num = 0
    for _ in range(num_infer_reqs):
        resp = resp_q.get()
        # print(f"Get response of infer request {infer_num}...")
        # print(json.dumps(resp, indent=4))

        if resp["status"] == "dropped":
            # Doesn't count dropped request for goodput
            latency = 9999 
        else:
            latency = resp["latency"]

        infer_result_writer.writerow([infer_num, resp["start_time"], resp["end_time"], resp["queuing_delay"], latency, resp["status"]])
        infer_num += 1
    infer_result_f.close()

    print("Shutdown gpu manager...")
    gpu_manager.do_shutdown = True

    # Get train info 
    resp = resp_q.get()
    gpu_manager.running_be_info = None
    # print(json.dumps(resp, indent=4))
    for start_time, end_time, duration in zip(resp["start_times"], resp["end_times"], resp["durations"]):
        train_result_writer.writerow([train_num, start_time, end_time, duration])
    train_result_f.close()

    gpu_manager_runner.join()
