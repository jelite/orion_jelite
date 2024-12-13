import sys
import glob
from argparse import ArgumentParser
import csv
import pandas as pd

argparser = ArgumentParser()
argparser.add_argument("--latency_bound", type=int, default=100)
argparser.add_argument("--total_iter", type=int, default=4320)

args = argparser.parse_args()

latency_bound = args.latency_bound
total_iter = args.total_iter

data_list = []
print("file_name, goodput, success, drop, fail")
for file_name in glob.glob(f"./*.txt"):
    if str(latency_bound) not in file_name:
        continue
    with open(file_name, 'r') as f:
        lines = f.readlines()
        goodput = 0
        success = 0
        drop = 0
        fail = 0
        log_file = file_name.replace("_withinfer.txt", "_total_withinfer.log")
        try:
            with open(log_file, 'r') as f:
                total_time = float(f.readline().strip().split(',')[0])
        except:
            print(f"no log file {log_file}")
            continue
            
        for line_num, line in enumerate(lines):
            data = line.split(",")
            if(data[0] == "passed"):
                drop += 1
            else:
                dur_with_queuing_delay = float(data[0])
                if(dur_with_queuing_delay > latency_bound):
                    fail += 1
                else:
                    success += 1    

        print(f"{file_name}, {success/total_time}, {success}, {drop}, {fail}, {total_time}")
