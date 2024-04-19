#!/bin/bash

base_rps=(59.13 85.05 320.43 41.52 31.5 64.83) #X2
# base_rps=(39.41 56.70 213.62 27.67 20.99 43.22) #X3

# base_rps=(29.56 42.53 160.22 20.75 15.75 46.01 32.42)
num_reqs=2000

for num in 0; do
    for rps in ${base_rps[@]}; do
        for rps_ratio in 1; do
            python3.8 arrival_interval_generator.py \
                --rps=$(echo $rps*$rps_ratio | bc) \
                --num_reqs=$num_reqs \
                --num=$num
        done
    done
done