#.py model_name batch gpu is_infer(0=false/1=true)
# /usr/bin/python3.8 run_ncu.py vit_l_16 1 3 1
# /usr/bin/python3.8 run_ncu.py vit_l_16 8 3 0

# /usr/bin/python3.8 run_ncu.py --model_name "vit_l_16" --batch_size 8 --gpu_num 3 
# /usr/bin/python3.8 run_ncu.py --model_name "vit_l_16" --batch_size 1 --gpu_num 3 --do_infer
# /usr/bin/python3.8 run_ncu.py --model_name "swin_b" --batch_size 8 --gpu_num 3 
# /usr/bin/python3.8 run_ncu.py --model_name "swin_b" --batch_size 1 --gpu_num 3 --do_infer

/usr/bin/python3.8 run_ncu.py --model_name "resnet50" --batch_size 4 --gpu_num 3 --do_infer
/usr/bin/python3.8 run_ncu.py --model_name "resnet50" --batch_size 1 --gpu_num 3 --do_infer
