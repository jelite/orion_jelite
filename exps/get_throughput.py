import glob

# for file in glob.glob("./train_0416/*.txt"):
    
#     through_sum = 0
#     num = 0

#     train = file.split("x")[0]
    
#     with open(file, 'r') as f:
#         lines = f.readlines()
#     data_num = 0
    
#     through_sum_list = []
#     through_sum_list.append({})
#     through_sum_list[-1]["through_sum"] = 0
#     through_sum_list[-1]["data_num"] = 0

#     data_num = 0
#     data_list = []
#     for line in lines:
        
#         if (data_num != int(line.split(",")[-1])):
#             break
#             data_num = int(line.split(",")[-1])
#             through_sum_list.append({})
#             through_sum_list[-1]["through_sum"] = 0
#             through_sum_list[-1]["data_num"] = 0

#         data = float(line.split(",")[0])
#         data_list.append(data)
        
#         if "vit" in train or  "swin" in train:
#             through_sum_list[-1]["through_sum"] += 8/data
#             through_sum_list[-1]["data_num"] += 1
#         else:
#             through_sum_list[-1]["through_sum"] += 64/data
#             through_sum_list[-1]["data_num"] += 1

#     through_sum = 0
#     for t in through_sum_list:
#         through_sum += t["through_sum"]/t["data_num"]

#     # print(f"{file}, {through_sum/len(through_sum_list)}, {len(through_sum_list)}")
#     print(f"{file}, {sum(data_list)}, {through_sum_list[0]['data_num']}")

#NEW
for file in glob.glob("./train_0416/*.txt"):
    
    train0 ={"num":0, "dur":0}
    train1 ={"num":0, "dur":0}
    train = file.split("x")[0]
    bs = 0
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if int(line[-2]):
                train1["num"] += 1
            else:
                train0["num"] += 1

    goodput_file = f"./exp_data_0418/{file.split('/')[-1][:-4]}_total.log"
    with open(goodput_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if int(line[-2]):
                train1["dur"] = float(line.split(',')[0])
            else:
                train0["dur"] = float(line.split(',')[0])

    if "vit" in train or  "swin" in train:
        bs = 8
    else:
        bs = 64
    print(f'{file},{train0["dur"]},{train0["num"]},{(bs*train0["num"])/(train0["dur"]*1000)}')