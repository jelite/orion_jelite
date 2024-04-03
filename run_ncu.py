import time
import os
from argparse import ArgumentParser

argparser = ArgumentParser()
argparser.add_argument("--model_name", type=str)
argparser.add_argument("--batch_size", type=int)
argparser.add_argument("--gpu_num", type=int)
argparser.add_argument("--do_infer", action='store_true')
args = argparser.parse_args()
        
model_name = args.model_name
batch_size = args.batch_size
gpu_num = args.gpu_num
do_infer = args.do_infer

if not os.path.exists("/workspace/config_files"):
    os.system("mkdir /workspace/config_files")
if do_infer:
    save_dir = f"/workspace/config_files/{model_name}_b{batch_size}_infer"
else:
    save_dir = f"/workspace/config_files/{model_name}_b{batch_size}_train"
    save_additional_dir = f"/workspace/config_files/{model_name}_b{batch_size}_additional_train"

if not os.path.exists(save_dir):
    os.system(f"mkdir {save_dir}")
if not do_infer:
    if not os.path.exists(save_additional_dir):
        os.system(f"mkdir {save_additional_dir}")
    

start = time.time()

if do_infer:
    os.system(f'CUDA_VISIBLE_DEVICES={gpu_num} ncu -o\
            {save_dir}/output_ncu --set detailed --profile-from-start off \
            /usr/bin/python3.8 script.py --model_name {model_name} --batch_size {batch_size} --do_infer')
    
    os.system(f'CUDA_VISIBLE_DEVICES={gpu_num} ncu --csv --set detailed --profile-from-start off \
            /usr/bin/python3.8 script.py --model_name {model_name} --batch_size {batch_size} --do_infer > {save_dir}/output_ncu.csv')
else:
    os.system(f'CUDA_VISIBLE_DEVICES={gpu_num} ncu -o\
            {save_additional_dir}/output_ncu --set detailed --profile-from-start off\
            /usr/bin/python3.8 script.py --model_name {model_name} --batch_size {batch_size} --do_additional')
    
    os.system(f'CUDA_VISIBLE_DEVICES={gpu_num} ncu --csv --set detailed --profile-from-start off \
            /usr/bin/python3.8 script.py --model_name {model_name} --batch_size {batch_size} --do_additional > {save_additional_dir}/output_ncu.csv')
    
    os.system(f'CUDA_VISIBLE_DEVICES={gpu_num} ncu -o\
            {save_dir}/output_ncu --set detailed --profile-from-start off \
            /usr/bin/python3.8 script.py --model_name {model_name} --batch_size {batch_size}')
    
    os.system(f'CUDA_VISIBLE_DEVICES={gpu_num} ncu --csv --set detailed --profile-from-start off \
            /usr/bin/python3.8 script.py --model_name {model_name} --batch_size {batch_size} > {save_dir}/output_ncu.csv')
end = time.time()

with open(f"{save_dir}/total_duration_ncu.txt", 'w') as f:
    f.write(str(end-start))
    
print("Done")