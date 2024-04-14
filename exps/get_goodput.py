import sys
import glob
from argparse import ArgumentParser
import csv


argparser = ArgumentParser()
argparser.add_argument("--latency_bound", type=int)
# argparser.add_argument("--total_trial", type=int)

args = argparser.parse_args()

latency_bound = args.latency_bound

for file in glob.glob(f"./*.txt"):
    if str(latency_bound) not in file:
        continue
    infer = file.split("x")[1]
    infer = infer.split("_")[0]
    
    success = 0
    passed = 0
    with open(file, 'r') as f:
        lines = f.readlines()
    line_num = 0
    for line in lines:
        # if line_num > 1500:
            # break
        data = line.split(",")[0]
        if "passed" not in data :
            data = float(data)
            if data < latency_bound:
                success += 1
        else:
            passed += 1
        line_num =+ 1
    total_time = 0  
    with open(f"{file[:-4]}_total.log", 'r') as f:
        total_time = float(f.readline().split(",")[0])
        # print(f"{file}\t{total_time}\t{success}\t{passed}\t{2000-success-passed}")
        # print(f"{file},{total_time},{success},{passed},{2000-success-passed}")
    print(f"{file}\t{success/total_time}")
