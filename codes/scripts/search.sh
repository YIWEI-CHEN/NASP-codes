#!/usr/bin/env bash

GPU=2
BATCH_SIZE=72
SEED=2
EXP_PATH="exp/single/batch_size${BATCH_SIZE}_gpu${GPU}"
EPOCH=50

python train_search.py --data /home/yiwei/cifar10 --batch_size ${BATCH_SIZE} --gpu ${GPU} \
    --save ${EXP_PATH} --seed ${SEED} --exec_script scripts/search.sh --epochs ${EPOCH} \
    --worker 0