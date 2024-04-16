import glob


for file in glob.glob("./train/*.txt"):
    
    through_sum = 0
    num = 0

    train = file.split("x")[0]
    
    with open(file, 'r') as f:
        lines = f.readlines()
    data_num = 0
    
    through_sum_list = []
    through_sum_list.append({})
    through_sum_list[-1]["through_sum"] = 0
    through_sum_list[-1]["data_num"] = 0

    data_num = 0
    for line in lines:
        data = float(line.split(",")[0])
        if (data_num != int(line.split(",")[-1])):
            break
            data_num = int(line.split(",")[-1])
            through_sum_list.append({})
            through_sum_list[-1]["through_sum"] = 0
            through_sum_list[-1]["data_num"] = 0

        # print(data_num)
        if "vit" in train or  "swin" in train:
            through_sum_list[-1]["through_sum"] += 8/data
            through_sum_list[-1]["data_num"] += 1
        else:
            through_sum_list[-1]["through_sum"] += 64/data
            through_sum_list[-1]["data_num"] += 1

    through_sum = 0
    for t in through_sum_list:
        through_sum += t["through_sum"]/t["data_num"]

    print(f"{file}, {through_sum/len(through_sum_list)}, {len(through_sum_list)}")
    # print(f"{file},{through_sum/num}, {data_num+1}")