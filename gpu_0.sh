#.py model_name batch gpu is_infer(0=false/1=true)
# /usr/bin/python3.8 run_ncu.py densenet121 64 0 0
# /usr/bin/python3.8 run_ncu.py resnet50 64 0 0
# /usr/bin/python3.8 run_ncu.py densenet121 8 0 1

# /usr/bin/python3.8 run_ncu.py resnet50 8 0 1

/usr/bin/python3.8 run_ncu.py --model_name "mnasnet1_3" --batch_size 8 --gpu_num 0 --do_infer

