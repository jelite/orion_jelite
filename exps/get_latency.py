import glob

for file in glob.glob("*.txt"):

    infer = file.split("x")[1]
    infer = infer.split("_")[0]

    kernel_latency_sum = []
    queuing_delay_sum = []

    with open(file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if "passed" in line:
            continue
        data = float(line.split(",")[1])
        kernel_latency_sum.append(data)
        
        data = float(line.split(",")[2])
        queuing_delay_sum.append(data)

    print(f"{file} {sum(kernel_latency_sum)/len(kernel_latency_sum)} {sum(queuing_delay_sum)/len(queuing_delay_sum)}")
