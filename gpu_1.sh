# /usr/bin/python3.8 run_ncu.py --model_name "mobilenet_v3_large" --batch_size 64 --gpu_num 1 
# /usr/bin/python3.8 run_ncu.py --model_name "mobilenet_v3_large" --batch_size 8 --gpu_num 1 --do_infer

# /usr/bin/python3.8 run_ncu.py --model_name "efficientnet_v2_m" --batch_size 64 --gpu_num 1 
/usr/bin/python3.8 run_ncu.py --model_name "efficientnet_v2_m" --batch_size 4 --gpu_num 1 --do_infer
/usr/bin/python3.8 run_ncu.py --model_name "efficientnet_v2_m" --batch_size 1 --gpu_num 1 --do_infer
# /usr/bin/python3.8 run_ncu.py -swin_b 8 1 0
/usr/bin/python3.8 run_ncu.py --model_name "mobilenet_v3_large" --batch_size 4 --gpu_num 1 --do_infer
# /usr/bin/python3.8 run_ncu.py swin_b 1 1 1
/usr/bin/python3.8 run_ncu.py --model_name "mobilenet_v3_large" --batch_size 1 --gpu_num 1 --do_infer
