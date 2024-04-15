#!/bin/bash

train_batch=(64)
infer_batch=(8)
train=("efficientnet_v2_m" "mobilenet_v3_large" "resnet50" "swin_b" "vit_l_16")
infer=("densenet121" "efficientnet_v2_m" "mobilenet_v3_large" "resnet50" "swin_b" "vit_l_16")
base_rps=(19.71 13.84 106.81 28.35 21.61 10.50)
dist_method=("poisson")

for train_bs in ${train_batch[@]}; do
    for infer_bs in ${infer_batch[@]}; do
        for bound in 100.0000 200.0000; do
            for train_model in ${train[@]}; do
                for i in "${!infer[@]}"; do
                    for rps_ratio in 1 3; do
                        for dist in ${dist_method[@]}; do
                            python overall_test_analyzer.py \
                                --exp=bound${bound}-${train_model}-b${train_bs}-CrossEntropyLoss-SGDX${infer[$i]}-b${infer_bs}-${dist}-rps$(echo "${base_rps[$i]}"*$rps_ratio | bc)
                                # --type=mps \
                        done
                    done
                done
            done
        done
    done
done
