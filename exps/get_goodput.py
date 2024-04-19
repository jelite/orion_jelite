import sys
import glob
from argparse import ArgumentParser
import csv
import pandas as pd

argparser = ArgumentParser()
argparser.add_argument("--latency_bound", type=int, default=50)
argparser.add_argument("--total_iter", type=int, default=2000)

args = argparser.parse_args()

latency_bound = args.latency_bound
total_iter = args.total_iter

# print("file,trial,goodput,success,drop,fail")

data_list = []
for file_name in glob.glob(f"./*.txt"):
    if str(latency_bound) not in file_name:
        continue
    with open(file_name, 'r') as f:
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
       
            # if line_num < 2000:
            #     l = line.split(",")
                # if "passed" not in line.split(",")[0] :
                    # print(f"{l[0]},{l[1]},{l[2]},{float(l[0]) < latency_bound}")
                # else:
                    # print(f"{l[0]},{l[1]},{l[2][:-1]},{False}")

        trial_num = len(goodput_list)
        err = False
        try:
            with open(f"{'_'.join(file_name.split('_')[:-1])}_total_withinfer.log",'r') as f:
            # with open(f"{file_name[:-4]}_total.log", 'r') as f:
                for i in range(trial_num):
                    total_time_str = f.readline().split(",")[0]
                    # print(file, total_time_str)
                    total_time =  float(total_time_str)
                    goodput_list[i]["total_time"] = total_time
        except FileNotFoundError:
            err = True
            # Handle the error: log an error message or take corrective action
            print(f"Error: The file '{file_name}' does not exist!")
        
        if (not err):
            goodputs = []
            for i in range(trial_num):
                success = goodput_list[i]["success"]
                drop = goodput_list[i]["drop"]
                fail = total_iter - success - drop
                total_time = goodput_list[i]["total_time"]
                pure_time = pure_list[i]["pure_time"]
                file_name = file_name.split('/')[-1]
                goodputs.append(success/total_time)
                # print(f"{file_name},Trial-{i},{success/total_time},{success},{drop},{fail},{total_time},{pure_time/(success+fail)}")
                data_list.append((file_name,success/total_time,success,drop,fail,total_time))
    # if (not err):
        # print(f"{file_name}, {sum(goodputs)/len(goodputs)}, {len(goodputs)}")
df = pd.DataFrame(data_list, columns=["file", "goodput", "success", "drop", "fail", "total_time"])
df =df.sort_values("file")
print(df)