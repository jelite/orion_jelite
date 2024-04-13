#!/bin/bash

train_batch=(64)
infer_batch=(8)
# train=("densenet121" "resnet50" "mobilenet_v3_large" "efficientnet_v2_m" "vit_l_16" "swin_b")
# infer=("densenet121" "resnet50" "mobilenet_v3_large" "efficientnet_v2_m" "vit_l_16" "swin_b")
# base_rps=(19.71 28.35 106.81 13.84 10.50 21.61)
train=("densenet121")
infer=("resnet50")
base_rps=(28.35)
dist_method=("poisson")

/workspace/gmanager/gmanager/mps_on.sh 0
for train_bs in ${train_batch[@]}; do
    for infer_bs in ${infer_batch[@]}; do
        for train_model in ${train[@]}; do
            for i in "${!infer[@]}"; do
                for bound in 200; do
                    for rps_ratio in 1; do
                        for dist in ${dist_method[@]}; do
                            for exp_num in 0 1 2; do
                                export GMANAGER_LATENCY_BOUND=$bound
                                export GMANAGER_TENSORRT=1
                                export GMANAGER_CUDA_GRAPH=1
                                export GMANAGER_DROP=1
                                export GMANAGER_PAUSE=1
                                export GMANAGER_PAUSE_INTERVAL_MS=0.05
                                export GMANAGER_BATCH_SPLIT=0

                                export CUDA_VISIBLE_DEVICES=0
                                nice -n -20 python gmanager_test.py \
                                    --gpu=0 \
                                    --train=$train_model \
                                    --train-bs=$train_bs \
                                    --infer="${infer[$i]}" \
                                    --infer-bs=$infer_bs \
                                    --dist=$dist \
                                    --rps=$(echo "${base_rps[$i]}"*$rps_ratio | bc) \
                                    --exp-num=$exp_num
                                unset CUDA_VISIBLE_DEVICES

                                unset GMANAGER_LATENCY_BOUND
                                unset GMANAGER_TENSORRT
                                unset GMANAGER_CUDA_GRAPH
                                unset GMANAGER_DROP
                                unset GMANAGER_PAUSE
                                unset GMANAGER_PAUSE_INTERVAL_MS
                                unset GMANAGER_BATCH_SPLIT
                            done
                        done
                    done
                done
            done
        done
    done
done
/workspace/gmanager/gmanager/mps_off.sh 0

# train=("densenet121")
# infer=("resnet50")
# base_rps=(28.35)
# dist_method=("poisson")
# 
# /workspace/gmanager/gmanager/mps_on.sh 0
# for train_bs in ${train_batch[@]}; do
#     for infer_bs in ${infer_batch[@]}; do
#         for train_model in ${train[@]}; do
#             for i in "${!infer[@]}"; do
#                 for bound in 200; do
#                     for rps_ratio in 1; do
#                         for dist in ${dist_method[@]}; do
#                             for exp_num in 0 1 2; do
#                                 export GMANAGER_LATENCY_BOUND=$bound
#                                 export GMANAGER_TENSORRT=1
#                                 export GMANAGER_CUDA_GRAPH=1
#                                 export GMANAGER_DROP=0
#                                 export GMANAGER_PAUSE=0
#                                 export GMANAGER_PAUSE_INTERVAL_MS=0
#                                 export GMANAGER_BATCH_SPLIT=0
# 
#                                 export CUDA_VISIBLE_DEVICES=0
#                                 nice -n -20 python gmanager_test.py \
#                                     --gpu=0 \
#                                     --train=$train_model \
#                                     --train-bs=$train_bs \
#                                     --infer="${infer[$i]}" \
#                                     --infer-bs=$infer_bs \
#                                     --dist=$dist \
#                                     --rps=$(echo "${base_rps[$i]}"*$rps_ratio | bc) \
#                                     --exp-num=$exp_num
#                                 unset CUDA_VISIBLE_DEVICES
# 
#                                 unset GMANAGER_LATENCY_BOUND
#                                 unset GMANAGER_TENSORRT
#                                 unset GMANAGER_CUDA_GRAPH
#                                 unset GMANAGER_DROP
#                                 unset GMANAGER_PAUSE
#                                 unset GMANAGER_PAUSE_INTERVAL_MS
#                                 unset GMANAGER_BATCH_SPLIT
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done
# /workspace/gmanager/gmanager/mps_off.sh 0
