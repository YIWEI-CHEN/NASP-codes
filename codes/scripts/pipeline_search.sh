#!/usr/bin/env bash

GPU=6,7
BATCH_SIZE=68
SEED=2
EPOCH=50
MICRO_BATCH_RATIO=0.5
EXP_PATH="exp/pipeline/batch_size${BATCH_SIZE}_micro${MICRO_BATCH_RATIO}_gpu${GPU}"

python train_search.py --data /home/yiwei/cifar10 --batch_size ${BATCH_SIZE} --gpu ${GPU} \
    --save ${EXP_PATH} --seed ${SEED} --exec_script scripts/pipeline_search.sh --epochs ${EPOCH} \
    --worker 0 --micro_batch_ratio ${MICRO_BATCH_RATIO}