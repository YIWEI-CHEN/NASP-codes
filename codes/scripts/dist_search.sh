#!/usr/bin/env bash

GPU="0,1,2,3"
TRAIN_BATCH_SIZE=272
VALID_BATCH_SIZE=272
SEED=2
PORT=50019
EPOCH=50
EXP_PATH="exp/distributed/train_bz${TRAIN_BATCH_SIZE}_valid_bz${VALID_BATCH_SIZE}_gpu${GPU}"

python train_search.py --data /home/yiwei/cifar10 --gpu ${GPU} \
    --save ${EXP_PATH} --seed ${SEED} --exec_script scripts/dist_search.sh \
    --world-size 1 --rank 0 --workers 0 --epochs ${EPOCH} \
    --dist-url "tcp://datalab3.engr.tamu.edu:${PORT}" \
    --train_batch_size ${TRAIN_BATCH_SIZE} --valid_batch_size ${VALID_BATCH_SIZE} \
    --subproblem_maximum_iterations 1