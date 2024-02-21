import time
import os
import sys 

model_name = sys.argv[1]
batch_size = int(sys.argv[2])
gpu_num = int(sys.argv[3])
# do_eval = bool(sys.argv[3])

start = time.time()
save_dir = f"/workspace/{model_name}_b{batch_size}_train"
os.system(f'CUDA_VISIBLE_DEVICES={gpu_num} ncu --csv --set detailed --nvtx --nvtx-include "start/" /usr/bin/python3.8 script.py {model_name} {batch_size} 0 ncu > {save_dir}/output_ncu.csv')
end = time.time()

# with open(f"{save_dir}/duration_ncu.txt", 'w') as f:
#     f.write(str(end-start))