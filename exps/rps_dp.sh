#!/bin/bash

base_rps=(44.03 44.03 46.46 43.98 45.64) # 4 gpus
# base_rps=(88.695 480.645 47.25) # 2 gpus
num_reqs=1580

for num in 0; do
    for rps in ${base_rps[@]}; do
        /usr/bin/python3.8 rps_dp.py \
            --rps=$(echo $rps) \
            --num_reqs=$num_reqs \
            --num=$num
    done
done