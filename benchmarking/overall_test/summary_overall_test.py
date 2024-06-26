import os
import glob
import argparse
import csv

import numpy as np
import parse

exclude_list = [
    "vit_l_16",
]


def get_results(dirpath, train_result_dict, infer_result_dict):
    os.chdir(dirpath)

    result_dirs = glob.glob("*")
    for result_dir in result_dirs:
        os.chdir(os.path.join(dirpath, result_dir))

        if os.stat("train-result.csv").st_size == 0 or os.stat("infer-result.csv").st_size == 0:
            print(os.getcwd())
            continue
            
        do_not_check = False
        for exclude_model in exclude_list:
            if exclude_model in result_dir:
                do_not_check = True
        if do_not_check:
            continue

        # bound10.8340-resnet50-b64-CrossEntropyLoss-AdamXmobilenet_v3_large-b8-poisson-rps120
        latency_bound, train_model, train_bs, loss_func, optimizer, infer_model, infer_bs, dist, rps = parse.parse("bound{}-{}-b{}-{}-{}X{}-b{}-{}-rps{}", result_dir).fixed
        latency_bound = float(latency_bound)
        train_bs = int(train_bs)
        infer_bs = int(infer_bs)
        rps = float(rps)

        train_model = f"{train_model}-b{train_bs}-{loss_func}-{optimizer}"
        infer_model = f"{infer_model}-b{infer_bs}"
        model_comb = f"{train_model}-b{train_bs}-{loss_func}-{optimizer}X{infer_model}-b{infer_bs}"

        if dist not in train_result_dict.keys():
            train_result_dict[dist] = {}
        if latency_bound not in train_result_dict[dist].keys():
            train_result_dict[dist][latency_bound] = {}
        if train_model not in train_result_dict[dist][latency_bound].keys():
            train_result_dict[dist][latency_bound][train_model] = {}
        if infer_model not in train_result_dict[dist][latency_bound][train_model].keys():
            train_result_dict[dist][latency_bound][train_model][infer_model] = {}
        if rps not in train_result_dict[dist][latency_bound][train_model][infer_model].keys():
            train_result_dict[dist][latency_bound][train_model][infer_model][rps] = []

        if dist not in infer_result_dict.keys():
            infer_result_dict[dist] = {}
        if latency_bound not in infer_result_dict[dist].keys():
            infer_result_dict[dist][latency_bound] = {}
        if train_model not in infer_result_dict[dist][latency_bound].keys():
            infer_result_dict[dist][latency_bound][train_model] = {}
        if infer_model not in infer_result_dict[dist][latency_bound][train_model].keys():
            infer_result_dict[dist][latency_bound][train_model][infer_model] = {}
        if rps not in infer_result_dict[dist][latency_bound][train_model][infer_model].keys():
            infer_result_dict[dist][latency_bound][train_model][infer_model][rps] = []

        train_result_file = open("train-result.csv", "r")
        train_result = list(csv.reader(train_result_file))
        train_result.pop(0)
        train_starts = np.array([float(start_time) for _, start_time, _, _ in train_result])
        train_ends = np.array([float(end_time) for _, _, end_time, _ in train_result])
        train_durations = [float(duration) for _, _, _, duration in train_result]

        infer_result_file = open("infer-result.csv", "r")
        infer_result = list(csv.reader(infer_result_file))
        infer_result.pop(0)
        infer_starts = np.array([float(start_time) for _, start_time, _, _, _, _ in infer_result])
        infer_ends = np.array([float(end_time) for _, _, end_time, _, _, _ in infer_result])
        total_infer_req_start_time = np.min(infer_starts)
        total_infer_req_end_time = np.max(infer_ends)
        infer_latencies = np.array([float(queuing_delay) + float(latency) for _, _, _, queuing_delay, latency, _ in infer_result])
        if args.metric == "latency":
            p99_infer_latency = np.percentile(infer_latencies, q=99, method="nearest")
            infer_result_dict[dist][latency_bound][train_model][infer_model][rps].append(p99_infer_latency)
        elif args.metric == "goodput":
            # goodput - metric req/s
            total_infer_req_process_dur = total_infer_req_end_time - total_infer_req_start_time
            goodput = np.count_nonzero(infer_latencies < latency_bound) / total_infer_req_process_dur
            infer_result_dict[dist][latency_bound][train_model][infer_model][rps].append(goodput)

        true_train_result_dict_start = np.where(train_starts < total_infer_req_start_time)[0][-1]
        true_train_result_dict_end = np.where(train_ends < total_infer_req_end_time)[0][-1]
        if true_train_result_dict_end - true_train_result_dict_start > 1:
            true_train_result_dict_end -= 1
        train_result_dict[dist][latency_bound][train_model][infer_model][rps].append(train_bs / np.mean(train_durations[true_train_result_dict_start:true_train_result_dict_end]))

        os.chdir("..")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-bs", type=int, default=64)
    parser.add_argument("--infer-bs", type=int, default=8)
    parser.add_argument("--metric", type=str, default="latency", choices=["latency", "goodput"])
    args = parser.parse_args()

    mps_result_dirname = os.path.join(os.path.expanduser("~"), ".results", f"mps-train-b{args.train_bs}Xinfer-b{args.infer_bs}")
    ours_result_dirname = os.path.join(os.path.expanduser("~"), ".results", f"ours-train-b{args.train_bs}Xinfer-b{args.infer_bs}")
    num_mps_result_dirs = len(glob.glob(f"{mps_result_dirname}*"))
    num_ours_result_dirs = len(glob.glob(f"{ours_result_dirname}*"))
    
    mps_train_results = {}
    mps_infer_results = {}
    ours_train_results = {}
    ours_infer_results = {}
    for i in range(num_ours_result_dirs):
        # if i > 1:
        #     break
        get_results(f"{mps_result_dirname}-{i}", mps_train_results, mps_infer_results)
        get_results(f"{ours_result_dirname}-{i}", ours_train_results, ours_infer_results)

    reordered_mps_infer_results = {}
    reordered_ours_infer_results = {}
    reordered_mps_train_results = {}
    reordered_ours_train_results = {}

    dists = list(ours_infer_results.keys())    
    for dist in dists:
        latency_bounds = sorted(list(ours_infer_results[dist].keys()))
        for latency_bound in latency_bounds:
            train_models = sorted(list(ours_infer_results[dist][latency_bound].keys()))
            for train_model in train_models:
                infer_models = sorted(list(ours_infer_results[dist][latency_bound][train_model].keys()))
                for infer_model in infer_models:
                    rps_list = sorted([rps for rps in ours_infer_results[dist][latency_bound][train_model][infer_model].keys()])
                    for idx, rps in enumerate(rps_list):
                        if train_model not in reordered_mps_infer_results.keys():
                            reordered_mps_infer_results[train_model] = {}
                        if infer_model not in reordered_mps_infer_results[train_model].keys():
                            reordered_mps_infer_results[train_model][infer_model] = {}
                        if rps not in reordered_mps_infer_results[train_model][infer_model].keys():
                            reordered_mps_infer_results[train_model][infer_model][rps] = {}

                        if train_model not in reordered_ours_infer_results.keys():
                            reordered_ours_infer_results[train_model] = {}
                        if infer_model not in reordered_ours_infer_results[train_model].keys():
                            reordered_ours_infer_results[train_model][infer_model] = {}
                        if rps not in reordered_ours_infer_results[train_model][infer_model].keys():
                            reordered_ours_infer_results[train_model][infer_model][rps] = {}

                        if train_model not in reordered_mps_train_results.keys():
                            reordered_mps_train_results[train_model] = {}
                        if infer_model not in reordered_mps_train_results[train_model].keys():
                            reordered_mps_train_results[train_model][infer_model] = {}
                        if rps not in reordered_mps_train_results[train_model][infer_model].keys():
                            reordered_mps_train_results[train_model][infer_model][rps] = {}

                        if train_model not in reordered_ours_train_results.keys():
                            reordered_ours_train_results[train_model] = {}
                        if infer_model not in reordered_ours_train_results[train_model].keys():
                            reordered_ours_train_results[train_model][infer_model] = {}
                        if rps not in reordered_ours_train_results[train_model][infer_model].keys():
                            reordered_ours_train_results[train_model][infer_model][rps] = {}
                        
                        if args.metric == "latency":
                            reordered_mps_infer_results[train_model][infer_model][rps][latency_bound] = f"{np.mean(mps_infer_results[dist][latency_bound][train_model][infer_model][rps]):>.4f}"
                            reordered_ours_infer_results[train_model][infer_model][rps][latency_bound] = f"{np.mean(ours_infer_results[dist][latency_bound][train_model][infer_model][rps]):>.4f}"
                        elif args.metric == "goodput":
                            reordered_mps_infer_results[train_model][infer_model][rps][latency_bound] = f"{np.mean(mps_infer_results[dist][latency_bound][train_model][infer_model][rps]) * 1000:>.4f}"
                            reordered_ours_infer_results[train_model][infer_model][rps][latency_bound] = f"{np.mean(ours_infer_results[dist][latency_bound][train_model][infer_model][rps]) * 1000:>.4f}"

                        reordered_mps_train_results[train_model][infer_model][rps][latency_bound] = f"{np.mean(mps_train_results[dist][latency_bound][train_model][infer_model][rps]):>.5f}"
                        reordered_ours_train_results[train_model][infer_model][rps][latency_bound] = f"{np.mean(ours_train_results[dist][latency_bound][train_model][infer_model][rps]):>.5f}"
    
    for train_model in reordered_ours_infer_results.keys():
        data = []
        for infer_model in reordered_ours_infer_results[train_model].keys():
            print(f"BE - {train_model} vs RT - {infer_model}")
            for rps in reordered_ours_infer_results[train_model][infer_model].keys():
                data.extend([v for _, v in reordered_mps_infer_results[train_model][infer_model][rps].items()])
        csv_line = ','.join(data)
        print(csv_line)
    print()

    # Orion
    print()

    for train_model in reordered_ours_infer_results.keys():
        data = []
        for infer_model in reordered_ours_infer_results[train_model].keys():
            print(f"BE - {train_model} vs RT - {infer_model}")
            for rps in reordered_ours_infer_results[train_model][infer_model].keys():
                data.extend([v for _, v in reordered_ours_infer_results[train_model][infer_model][rps].items()])
        csv_line = ','.join(data)
        print(csv_line)
    print()

    for train_model in reordered_ours_infer_results.keys():
        data = []
        for infer_model in reordered_ours_infer_results[train_model].keys():
            print(f"BE - {train_model} vs RT - {infer_model}")
            for rps in reordered_ours_infer_results[train_model][infer_model].keys():
                data.extend([v for _, v in reordered_mps_train_results[train_model][infer_model][rps].items()])
        csv_line = ','.join(data)
        print(csv_line)
    print()

    # Orion
    print()

    for train_model in reordered_ours_infer_results.keys():
        data = []
        for infer_model in reordered_ours_infer_results[train_model].keys():
            print(f"BE - {train_model} vs RT - {infer_model}")
            for rps in reordered_ours_infer_results[train_model][infer_model].keys():
                data.extend([v for _, v in reordered_ours_train_results[train_model][infer_model][rps].items()])
        csv_line = ','.join(data)
        print(csv_line)
    print()