import glob


for file in glob.glob("./train/*.txt"):
    
    through_sum = 0
    num = 0

    train = file.split("x")[0]
    
    with open(file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        data = float(line.split(",")[0])
        if "vit" in train or  "swin" in train:
            through_sum += 8/data
        else:
            through_sum += 64/data
        num+=1
        
    print(f"{file} : {through_sum/num}")