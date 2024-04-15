import sys
import glob
from argparse import ArgumentParser
import csv

argparser = ArgumentParser()
argparser.add_argument("--latency_bound", type=int)
argparser.add_argument("--total_iter", type=int)

args = argparser.parse_args()

latency_bound = args.latency_bound
total_iter = args.total_iter



for file in glob.glob(f"./exp_data/*.txt"):
    if str(latency_bound) not in file:
        continue
    if "swin_bxswin_b_64.83_" not in file:
        continue 
    with open(file, 'r') as f:
        lines = f.readlines()
        goodput_list = []
        pure_list = []
        a = 0
        for line_num, line in enumerate(lines):
            if(line_num % total_iter == 0):
                goodput_list.append({})
                pure_list.append({})
                goodput_list[-1]["success"] = 0
                goodput_list[-1]["drop"] = 0
                goodput_list[-1]["total_time"] = 0
                pure_list[-1]["pure_time"] = 0
                a = 0
            data = line.split(",")[0]
            if "passed" not in data :
                pure_time = line.split(",")[1]
                pure_list[-1]["pure_time"] += float(pure_time)

                data = float(data)
                if data < latency_bound:
                    goodput_list[-1]["success"] += 1
            else:
                goodput_list[-1]["drop"] += 1
            if line_num < 2000:
                l = line.split(",")
                if "passed" not in line.split(",")[0] :
                    print(f"{l[0]},{l[1]},{l[2]},{float(l[0]) < latency_bound}")
                else:
                    print(f"{l[0]},{l[1]},{l[2][:-1]},{False}")

        trial_num = len(goodput_list)

        with open(f"{file[:-4]}_total.log", 'r') as f:
            for i in range(trial_num):
                total_time_str = f.readline().split(",")[0]
                total_time =  float(total_time_str)
                goodput_list[i]["total_time"] = total_time
        
        goodputs = []
        for i in range(trial_num):
            success = goodput_list[i]["success"]
            drop = goodput_list[i]["drop"]
            fail = total_iter - success - drop
            total_time = goodput_list[i]["total_time"]
            pure_time = pure_list[i]["pure_time"]
            file_name = file.split('/')[-1]
            goodputs.append(success/total_time)
            # print(f"{file_name},Trial-{i},{success/total_time},{success},{drop},{fail},{total_time},{pure_time/(success+fail)}")
