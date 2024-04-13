import numpy as np
import parse

import csv
import os
import argparse
import glob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="ours")
    parser.add_argument("--exp", type=str)
    args = parser.parse_args()

    latency_bound, train_model, train_bs, _, _, infer_model, infer_bs, _, rps = parse.parse("bound{}-{}-b{}-{}-{}X{}-b{}-{}-rps{}", args.exp).fixed
    latency_bound = float(latency_bound)
    train_bs = int(train_bs)
    infer_bs = int(infer_bs)
    if "vit" in train_model or "swin" in train_model:
        train_bs = 64
    if "vit" in infer_model or "swin" in infer_model:
        infer_bs = 8

    if args.type == "ours":
        result_dirname = os.path.join(os.path.expanduser("~"), ".results", f"ours-train-b{train_bs}Xinfer-b{infer_bs}")
    else:
        result_dirname = os.path.join(os.path.expanduser("~"), ".results", f"mps-train-b{train_bs}Xinfer-b{infer_bs}")
    num_result_dirs = len(glob.glob(f"{result_dirname}*"))

    for i in range(num_result_dirs):
        os.chdir(os.path.join(result_dirname+f"-{i}", args.exp))

        infer_result_file = open("infer-result.csv", "r")
        infer_result = list(csv.reader(infer_result_file))
        infer_result.pop(0)

        infer_starts = np.array([float(start_time) for _, start_time, _, _, _, _ in infer_result])
        infer_ends = np.array([float(end_time) for _, _, end_time, _, _, _ in infer_result])
        infer_latencies = np.array([float(queuing_delay) + float(latency) for _, _, _, queuing_delay, latency, _ in infer_result])
        infer_latencies_no_delay = np.array([float(latency) for _, _, _, _, latency, _ in infer_result])
        infer_status = np.array([status for _, _, _, _, _, status in infer_result])
        num_success = len(infer_latencies[infer_latencies < latency_bound])
        num_drops = len(infer_status[infer_status == "dropped"])
        num_failed = len(infer_latencies) - num_success - num_drops
        infer_latencies_no_delay = infer_latencies_no_delay[infer_latencies_no_delay != 9999]

        total_infer_req_start_time = np.min(infer_starts)
        total_infer_req_end_time = np.max(infer_ends)
        total_time = total_infer_req_end_time - total_infer_req_start_time

        print(f"name : {os.path.join(result_dirname+f'-{i}', args.exp)}")
        print("============================ RT Summary ===========================")
        print(f"total infer processing time : {total_time:>.2f}ms")
        print(f"number of success : {num_success}")
        print(f"number of drops : {num_drops}")
        print(f"number of failed : {num_failed}")
        print(f"Min latency : {np.min(infer_latencies_no_delay)}")
        print(f"Max latency : {np.max(infer_latencies_no_delay)}")
        print(f"req throughput : {len(infer_latencies)/total_time * 1000:>.4f}")
        print(f"req goodput : {num_success/total_time * 1000:>.4f}")
        print()

        train_result_file = open("train-result.csv", "r")
        train_result = list(csv.reader(train_result_file))
        train_result.pop(0)

        train_durs = np.array([float(dur) for _, _, _, dur in train_result])
        print("============================ BE Summary ===========================")
        print(f"Throughput : {train_bs/np.mean(train_durs):>.4f} (image/msec)")
        print(f"Average duration : {np.mean(train_durs):>.4f}ms")
        print(f"Min duration : {np.min(train_durs):>.4f}ms")
        print(f"Max duration : {np.max(train_durs):>.4f}ms")
        print(f"Duration var / avg : {np.var(train_durs) / np.mean(train_durs):>.4f}")
        print()
        print()
