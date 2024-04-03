import os

# models = ["resnet50", "mobilenet_v3_large", "vit_l_16", "densenet121", "efficientnet_v2_m"]
models = ["resnet50", "mobilenet_v3_large", "vit_l_16"]
# alias = ["rnet", "mnet", "vit", "enet", "swin", "dnet"]
alias = ["rnet", "mnet", "vit"]

models = ["densenet121"]
# models = ["resnet50"]
alias = ["dnet"]
# alias = ["rnet"]

is_be_infer = True
    
for trial in range(5):
    for rps in [1,2]:
        for slo in [100, 200]:
            for train_idx, train in enumerate(models):
                for infer_idx, infer in enumerate(models):
                    if "mnet" in infer:
                        max_be_duration = 100000
                    else:
                        max_be_duration = 160000

                    running_pair = f"{alias[train_idx]}_{alias[infer_idx]}"
                    running_pair = f"rnet_mnet"
                    print(f"{running_pair} is running")
                    if is_be_infer:
                        file_path = f"config_files/be_infer/rps_level{rps}_{slo}ms/{running_pair}.json"
                    else:
                        file_path = f"config_files/rps_level{rps}_{slo}ms/{running_pair}.json"
                    # os.system(f"LD_PRELOAD='{os.path.expanduser( '~' )}/orion/src/cuda_capture/libinttemp.so' python3.8 \
                    #             ../benchmarking/launch_jobs.py --algo orion --config_file {file_path} --orion_max_be_duration {max_be_duration} \
                    #                 --do_save --trial {trial}")
                    os.system(f"LD_PRELOAD='{os.path.expanduser( '~' )}/orion/src/cuda_capture/libinttemp.so' python3.8 \
                                ../benchmarking/launch_jobs.py --algo orion --config_file {file_path} --orion_max_be_duration {max_be_duration} \
                                      --do_save --trial {trial}")
                
                    quit()
                    
print("Done")