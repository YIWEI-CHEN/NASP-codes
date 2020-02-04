#!/usr/bin/env bash

GPU="0,1"
BATCH_SIZE=108
SEED=42
PORT=50017
EXP_PATH="exp/distributed/batch_size${BATCH_SIZE}_gpu${GPU}"
EPOCH=50

python train_search.py --data /home/yiwei/cifar10 --batch_size ${BATCH_SIZE} --gpu ${GPU} \
    --save ${EXP_PATH} --seed ${SEED} --exec_script scripts/dist_search.sh \
    --world-size 1 --rank 0 --workers 0 --epochs ${EPOCH} \
    --dist-url "tcp://datalab.cse.tamu.edu:${PORT}"