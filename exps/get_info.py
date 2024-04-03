import glob

for file in glob.glob("*.txt"):
    infer = file.split("x")[1]
    infer = infer.split("_")[0]

    latency_bound = int(file.split("ms")[0][-3])*100
    success = 0
    drop = 0
    timeout = 0
    with open(file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        data = line.split(",")[0]
        if "passed" not in data :
            data = float(data)
            if data < latency_bound:
                success += 1
            else:
                timeout += 1
        else:
            drop += 1

    total_time = 0
    with open(f"{file[:-4]}_total.log", 'r') as f:
        total_time = float(f.readline())

    print(f"{file} {success} {drop} {timeout}")
