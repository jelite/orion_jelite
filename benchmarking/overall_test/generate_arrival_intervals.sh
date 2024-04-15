#!/bin/bash

# base_rps=(19.71 28.35 106.81 13.84 10.50 30.67 21.61)
base_rps=(29.56 42.53 160.22 20.75 15.75 46.01 32.42)
num_reqs=2000

for num in 0 1 2; do
    for rps in ${base_rps[@]}; do
        for rps_ratio in 1; do
            python arrival_interval_generator.py \
                --rps=$(echo $rps*$rps_ratio | bc) \
                --num_reqs=$num_reqs \
                --num=$num
        done
    done
done