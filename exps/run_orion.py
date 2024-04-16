import os

models_t = ["rnet", "mnet", "vit_b", "vit_l", "enet", "swin"]
models_i = ["rnet", "mnet", "vit_b", "vit_l", "enet", "swin", "dnet", "sqnet", "shnet", "mnnet"]
models_i = ["sqnet", "shnet", "mnnet"]

is_be_infer = False

for trial in range(1):
    for slo in [100, 200, 300]:
        for rps in [1, 2]:
            for train in models_t:
                for infer in models_i:
                    if infer in ["mnet", "sqnet", "shnet", "mnnet"] :
                        max_be_duration = 100000
                    if "vit" in infer:
                        max_be_duration = 320000
                    else:
                        max_be_duration = 160000
                    running_pair = f"{train}_{infer}"
                    print(f"{running_pair} {rps} {slo} is running")
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
        print(f"SLO{slo}ms rps_level{rps} Done")
                    
# for trial in range(5):
#     for rps in [1,2]:
#         for slo in [100, 200]:
#             for train_idx, train in enumerate(models):
#                 for infer_idx, infer in enumerate(models):
#                     if "mnet" in infer:
#                         max_be_duration = 100000
#                     else:
#                         max_be_duration = 160000

#                     running_pair = f"{alias[train_idx]}_{alias[infer_idx]}"
#                     running_pair = f"rnet_mnet"
#                     print(f"{running_pair} is running")
#                     if is_be_infer:
#                         file_path = f"config_files/be_infer/rps_level{rps}_{slo}ms/{running_pair}.json"
#                     else:
#                         file_path = f"config_files/rps_level{rps}_{slo}ms/{running_pair}.json"
#                     # os.system(f"LD_PRELOAD='{os.path.expanduser( '~' )}/orion/src/cuda_capture/libinttemp.so' python3.8 \
#                     #             ../benchmarking/launch_jobs.py --algo orion --config_file {file_path} --orion_max_be_duration {max_be_duration} \
#                     #                 --do_save --trial {trial}")
#                     os.system(f"LD_PRELOAD='{os.path.expanduser( '~' )}/orion/src/cuda_capture/libinttemp.so' python3.8 \
#                                 ../benchmarking/launch_jobs.py --algo orion --config_file {file_path} --orion_max_be_duration {max_be_duration} \
#                                       --do_save --trial {trial}")
                
#                     quit()
                    
print("Done")
