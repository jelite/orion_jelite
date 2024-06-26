import os
import time
MicroS = 1000
MiliS = 1000*MicroS
num_runs = 1
trace_files = [
    #("Mobilenet", "ResNet50", "mnet_rnet", 160000),
    ("ResNet50", "ResNet50", "rnet_rnet", 1600),
]
    # ("ResNet50", "ResNet50", "rnet_rnet", 160000),
    # ("ResNet50", "MobileNetV2", "rnet_mnet", 100000),
    # ("MobileNetV2", "ResNet50", "mnet_rnet", 160000),
    # ("MobileNetV2", "MobileNetV2", "mnet_mnet", 100000),
    # ("ResNet101", "ResNet50", "rnet101_rnet", 160000),
    # ("ResNet101", "MobileNetV2", "rnet101_mnet", 100000),
    # ("BERT", "ResNet50", "bert_rnet", 160000),
    # ("BERT", "MobileNetV2", "bert_mnet", 100000),
    # ("Transformer", "ResNet50", "trans_rnet", 160000),
    # ("Transformer", "MobileNetV2", "trans_mnet", 100000),
for (be, hp, f, max_be_duration) in trace_files:
    for run in range(num_runs):
        print(be, hp, run, flush=True)
        # run
        print(f"################  {os.path.expanduser( '~' )}/orion/src/cuda_capture/libinttemp.so")
        file_path = f"config_files/{f}.json"
        os.system(f"LD_PRELOAD='{os.path.expanduser( '~' )}/orion/src/cuda_capture/libinttemp.so' python3.8 ../../benchmarking/launch_jobs.py --algo orion --config_file {file_path} --orion_max_be_duration {max_be_duration}")

        # copy results
        # os.system(f"cp client_1.json results/orion/{be}_{hp}_{run}_hp.json")
        # os.system(f"cp client_0.json results/orion/{be}_{hp}_{run}_be.json")

        # os.system("rm client_1.json")
        # os.system("rm client_0.json")
