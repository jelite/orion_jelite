import json
import csv

def get_kernel_num(file_name):
    num_rows = 0
    for row in open(file_name):
        num_rows += 1
    return num_rows-1

# models = ["resnet50", "mobilenet_v3_large", "efficientnet_v2_m", \
#           "vit_l_16", "swin_b", "densenet121"]
train_models = ["resnet50", "mobilenet_v3_large", "vit_b_16", "vit_l_16", "efficientnet_v2_m", "swin_b", "densenet121"]
infer_models = ["resnet50", "mobilenet_v3_large", "vit_b_16", "vit_l_16", "efficientnet_v2_m", "swin_b", "densenet121", "squeezenet1_1", "shufflenet_v2_x2_0", "mnasnet1_3"]
# alias = ["rnet", "mnet", "enet", "vit", "swin", "dnet"]
alias = ["rnet", "mnet", "vit_b", "vit_l", "enet", "swin", "dnet", "sqnet", "shnet", "mnnet"]


#for train x infer
for train_idx, train_name in enumerate(train_models):
    for infer_idx, infer_name in enumerate(infer_models):
        if "vit" in train_name or "swin" in train_name:
            train_batch = 8
        else:
            train_batch = 64
        if "vit" in infer_name or "swin" in infer_name:
            infer_batch = 1
        else:
            infer_batch = 8
        # [19.71, 28.35, 106.81, 13.84, 10.5, 21.61, 30.67]
        levels = [[29.56, 42.53, 160.22, 20.75, 15.75, 32.42, 46.01],[59.13, 85.05, 320.43, 41.52, 31.5, 64.83, 92.01]]
        latency_bounds = [100, 200, 300]
        for idx, level in enumerate(levels):
            for latency_bound in latency_bounds:
                if "dense" in infer_name:
                    rps = level[0]
                elif "resnet" in infer_name:
                    rps = level[1]
                elif "mobilenet" in infer_name:
                    rps = level[2] 
                elif "effi" in infer_name:
                    rps = level[3]  
                elif "vit_l" in infer_name:
                    rps = level[4]
                elif "swin" in infer_name:
                    rps = level[5]
                elif "vit_b" in infer_name:
                    rps = level[6]
                else:
                    rps = level[2]
                    
                train_kernel_file = f"/workspace/exps/kernel_files/{train_name}_b{train_batch}_train"
                train_additional_kernel_file = f"/workspace/exps/kernel_files/{train_name}_b{train_batch}_additional_train"
                infer_kernel_file = f"/workspace/exps/kernel_files/{infer_name}_b{infer_batch}_infer"
                
                configs = [
                        {
                        "arch": train_name,
                        "kernel_file": train_kernel_file,
                        "additional_kernel_file": train_additional_kernel_file,
                        "num_kernels": get_kernel_num(train_kernel_file),
                        "additional_num_kernels": get_kernel_num(train_additional_kernel_file),
                        "num_iters": 1100,
                        "args": {
                            "model_name": train_name,
                            "batchsize": train_batch,
                            "latency_bound": None,
                            "rps": 0,
                            "uniform": False,
                            "dummy_data": True,
                            "train": "train"
                        }
                    },
                        {
                        "arch": infer_name,
                        "kernel_file": infer_kernel_file,
                        "num_kernels": get_kernel_num(infer_kernel_file),
                        "num_iters": 2200,
                        "args": {
                            "model_name": infer_name,
                            "batchsize": infer_batch,
                            "rps": rps,
                            "latency_bound": latency_bound,
                            "uniform": False,
                            "dummy_data": True,
                            "train": "infer"
                        }
                    }
                ]
                with open(f"rps_level{idx+1}_{latency_bound}ms/{alias[train_idx]}_{alias[infer_idx]}.json", "w") as outfile:
                    json.dump(configs, outfile, indent=4)



# #for infer x infer
for train_idx, train_name in enumerate(infer_models):
    for infer_idx, infer_name in enumerate(infer_models):
        if "vit" in train_name or "swin" in train_name:
            train_batch = 1
        else:
            train_batch = 8
        if "vit" in infer_name or "swin" in infer_name:
            infer_batch = 1
        else:
            infer_batch = 8
        
        levels = [[29.56, 42.53, 160.22, 20.75, 15.75, 32.42, 46.01],[59.13, 85.05, 320.43, 41.52, 31.5, 64.83, 92.01]]
        latency_bounds = [100, 200, 300]
        for idx, level in enumerate(levels):
            for latency_bound in latency_bounds:
                if "dense" in infer_name:
                    rps = level[0]
                elif "resnet" in infer_name:
                    rps = level[1]
                elif "mobilenet" in infer_name:
                    rps = level[2] 
                elif "effi" in infer_name:
                    rps = level[3]  
                elif "vit_l" in infer_name:
                    rps = level[4]
                elif "swin" in infer_name:
                    rps = level[5]
                elif "vit_b" in infer_name:
                    rps = level[6]
                else:
                    rps = level[2]
                    
                    
                train_kernel_file = f"/workspace/exps/kernel_files/{train_name}_b{train_batch}_infer"
                infer_kernel_file = f"/workspace/exps/kernel_files/{infer_name}_b{infer_batch}_infer"
                
                configs = [
                        {
                        "arch": train_name,
                        "kernel_file": train_kernel_file,
                        "num_kernels": get_kernel_num(train_kernel_file),
                        "num_iters":5000,
                        "args": {
                            "model_name": train_name,
                            "batchsize": train_batch,
                            "latency_bound": None,
                            "rps": 0,
                            "uniform": False,
                            "dummy_data": True,
                            "train": "be_infer"
                        }
                    },
                        {
                        "arch": infer_name,
                        "kernel_file": infer_kernel_file,
                        "num_kernels": get_kernel_num(infer_kernel_file),
                        "num_iters":2200,
                        "args": {
                            "model_name": infer_name,
                            "batchsize": infer_batch,
                            "rps": rps,
                            "latency_bound": latency_bound,
                            "uniform": False,
                            "dummy_data": True,
                            "train": "infer"
                        }
                    }
                ]
                
                # configs = configs.values()
                with open(f"be_infer/rps_level{idx+1}_{latency_bound}ms/{alias[train_idx]}_{alias[infer_idx]}.json", "w") as outfile:
                    json.dump(configs, outfile, indent=4)

print("Done")
#2200
