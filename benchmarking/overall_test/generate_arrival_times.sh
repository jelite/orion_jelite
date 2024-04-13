#!/bin/bash

base_rps=(19.71 28.35 106.81 13.84 10.50 21.61)
num_reqs=2000

for num in 0 1 2; do
    for rps in ${base_rps[@]}; do
        for rps_ratio in 1 3; do
            python arrival_time_generator.py \
                --rps=$(echo $rps*$rps_ratio | bc) \
                --num_reqs=$num_reqs \
                --num=$num
        done
    done
done