import numpy as np

from argparse import ArgumentParser
from collections import OrderedDict
import json
import time
import csv
import os
import itertools

np.random.seed(time.time_ns() % 1000000000)


def gen_intervals_from_azure_v1_trace(trace_dir, 
                                      num_reqs_scale=1, 
                                      rps_scale=1):
    task_traces_fname = 'azure_v1_top_task_traces.json'
    task_traces_path = os.path.join(trace_dir, task_traces_fname)

    if not os.path.exists(task_traces_path):
        task_traces = OrderedDict()
        
        data_paths = [os.path.join(trace_dir, f"invocations_per_function_md.anon.d{day:>02}.csv") for day in range(1, 15)]
        for idx, data_path in enumerate(data_paths):
            print(data_path)

            data_file = open(data_path, 'r', newline="")
            csv_rdr = csv.DictReader(data_file)
            for row in csv_rdr:
                function_name = row["HashFunction"]
                hist_1min = np.array([int(row[str(j)]) for j in range(1, 1441)], dtype=np.int32)
                if idx+1 == 1:
                    assert function_name not in task_traces.keys()
                    task_traces[function_name] = hist_1min
                else:
                    expected_size = 1440 * idx
                    if function_name in task_traces.keys():
                        cur_size = task_traces[function_name].size
                        if cur_size != expected_size:
                            diff = expected_size - cur_size
                            assert diff % 1440 == 0
                            task_traces[function_name] = np.concatenate((task_traces[function_name], np.zeros((diff,), dtype=np.int32), hist_1min))
                        else:
                            task_traces[function_name] = np.concatenate((task_traces[function_name], hist_1min))
                    else:
                        task_traces[function_name] = np.concatenate((np.zeros((expected_size,), dtype=np.int32), hist_1min))

        for function_name, hist_1min in task_traces.items():
            if hist_1min.size != 14 * 1440:
                diff = 14 * 1440 - hist_1min.size
                assert diff % 1440 == 0
                task_traces[function_name] = np.concatenate((task_traces[function_name], np.zeros((diff,), dtype=np.int32)))
        
        picked_traces = {}
        for k, v in task_traces.items():
            if v.min() > v.max() * 0.1:
                picked_traces[k] = v.tolist()

        top_5_traces = dict(
            sorted(picked_traces.items(),
                key=lambda item: np.sum(item[-1]),
                reverse=True)[:5]
        )
        
        with open(task_traces_path, 'w') as f:
            json.dump(top_5_traces, f, indent=4)
    else:
        model_to_middle_rps = {
            'densenet121': 29.56,
            'renset50': 42.53,
            'mobilenet_v3_large': 160.22,
            'efficientnet_v2_m': 20.75,
            'vit': 15.75,
            'swin_b': 32.42,
        }

        task_traces_file = open(task_traces_path, 'r')
        data = json.load(task_traces_file)
        
        num_reqs_scale = 1/1000. # Maybe this is maximum
        for k, v in data.items():
            one_day_length = 1440
            rps_scale = model_to_middle_rps["densenet121"] / (np.mean(v) / 60)
            trace = np.array(v[:one_day_length])
            intervals = []
            for i, hist_1min in enumerate(trace):
                rps = hist_1min / 60 * rps_scale
                num_reqs = int(hist_1min * num_reqs_scale)
                intervals_1min = np.random.exponential(scale=1/rps, size=num_reqs-1).tolist()
                intervals.extend(intervals_1min)
                
                if i % 100 == 0:
                    print(f"[{i}] rps : {hist_1min / 60:>.2f} -> {rps:>.2f}")
                    print(f"[{i}] num_reqs : {hist_1min} -> {int(hist_1min * num_reqs_scale)}")

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--dist", type=str, default='poisson')
    argparser.add_argument("--rps", type=float, default=19.71)
    argparser.add_argument("--num_reqs", type=int, default=2000)
    argparser.add_argument("--num", type=int, default=0)
    args = argparser.parse_args()

    if args.dist == 'poisson':
        # For poisson dist
        fname = f'./traces-rps{args.rps}-reqs{args.num_reqs}-num{args.num}.json'
        arrival_times = np.random.exponential(scale=1/args.rps, size=args.num_reqs-1).tolist()
        with open(fname, "w") as f:
            json.dump(arrival_times, f, indent=4)
    elif args.dist == 'azure_v1':
        # For Azure v1
        gen_intervals_from_azure_v1_trace(os.path.join(os.getcwd(), 'arrival_intervals/azure_v1'))