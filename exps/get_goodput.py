import sys
import glob
from argparse import ArgumentParser
import csv


argparser = ArgumentParser()
argparser.add_argument("--latency_bound", type=int)
# argparser.add_argument("--total_trial", type=int)

args = argparser.parse_args()

latency_bound = args.latency_bound

for file in glob.glob(f"exp_data/*.txt"):
    if str(latency_bound) not in file:
        continue
    infer = file.split("x")[1]
    infer = infer.split("_")[0]
    
    success = 0
    
    with open(file, 'r') as f:
        lines = f.readlines()
    line_num = 0
    for line in lines:
        if line_num > 1500:
            break
        data = line.split(",")[0]
        if "passed" not in data :
            data = float(data)
            if data < latency_bound:
                success += 1
        line_num =+ 1
    total_time = 0  
    with open(f"{file[:-4]}_total.log", 'r') as f:
        total_time = float(f.readline().split(",")[0])
        print(f"total {total_time}, good {success}")
    print(f"{file} {success/total_time*3}")
