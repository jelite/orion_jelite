import os
import json
import numpy as np
import pandas as pd
import pdb
# model_to_high_rps = {
#     'densenet121': 59.13,
#     'resnet50': 85.05,
#     'mobilenet_v3_large': 320.43,
#     'efficientnet_v2_m': 41.52,
#     'vi_l_16': 31.5,
#     'swin_b': 64.84,
# }

model_to_middle_rps = {
    'densenet121': 29.56,
    'resnet50': 42.53,
    'mobilenet_v3_large': 160.22,
    'efficientnet_v2_m': 20.75,
    'vi_l_16': 15.75,
    'swin_b': 32.42,
}

def get_intervals(trace_type, rt_model):
    

    trace_root_dir = os.path.join(os.getcwd(), "arrival_intervals")


    one_day_length = 1440
    num_reqs_scale = 1/1000. # Maybe this is maximum
    # trace_path = os.path.join(trace_root_dir, "azure_v1", "azure_v1_top_task_traces.json")
    with open("./azure_v1_top_task_traces.json", "r") as f:
        traces = json.load(f)

    trace = np.array(list(traces.values())[3])
    rps_scale = model_to_middle_rps[rt_model] / (trace / 60).mean()
    intervals = []
    for i, hist_1min in enumerate(trace[:1440]):
        rps = hist_1min / 60 * rps_scale
        num_reqs = 1  #int(hist_1min * num_reqs_scale)
        intervals_1min = np.random.exponential(scale=1/rps, size=num_reqs).tolist()
        intervals.extend(intervals_1min)
        
        print(f"[{i}] rps : {hist_1min / 60} -> {rps}")
        print(f"[{i}] num_reqs : {hist_1min} -> {int(hist_1min * num_reqs_scale)}")

    return intervals

num_infer_reqs = 5000
models = list(model_to_middle_rps.keys())
for model in models:
    intervals = get_intervals("azure_v1", model)
    df = pd.DataFrame(intervals, columns=['interval'])
    df.to_csv(f"./test_rps.csv", index=False)

