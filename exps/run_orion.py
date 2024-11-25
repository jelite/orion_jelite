import os

models_t = ["vit_l", "rnet", "mnet", "enet", "swin"]
models_t = ["vit_l"]
# models_t = ["vit_l"]
# models_t = ["swin"]
models_i = ["vit_l"]
# models_i = ["dnet", "rnet", "mnet", "enet"]
# models_i = ["rnet", "mnet", "vit_l", "enet", "swin", "dnet"]
# models_i = ["sqnet", "shnet", "mnnet"]

is_be_infer = True
# for trial in range(1):
#     for slo in [100, 200, 300]:
#         for rps in [1, 2]:
#             for train in models_t:
#                 for infer in models_i:
#                     if infer in ["mnet", "sqnet", "shnet", "mnnet"] :
#                         max_be_duration = 100000
#                     if "vit" in infer:
#                         max_be_duration = 320000
#                     else:
#                         max_be_duration = 160000
#                     running_pair = f"{train}_{infer}"
#                     print(f"{running_pair} {rps} {slo} is running")
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
#         print(f"SLO{slo}ms rps_level{rps} Done")
                    

for trial in range(1):
    for slo in [50]:
        for rps in [1]:
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
                    
print("Done")
