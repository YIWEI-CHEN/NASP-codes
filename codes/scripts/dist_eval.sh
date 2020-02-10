#!/usr/bin/env bash

GPU="1,2"
TRAIN_BATCH_SIZE=272
VALID_BATCH_SIZE=64
SEED=0
PORT=50017
EXP_PATH="exp/distributed/train_bz${TRAIN_BATCH_SIZE}_valid_bz${VALID_BATCH_SIZE}_gpu${GPU}"
EPOCH=600
ARCH="SINGLE_NASP"

python dist_train.py --data /home/yiwei/cifar10 --gpu ${GPU} --save ${EXP_PATH} \
    --seed ${SEED} --cutout --auxiliary --exec_script scripts/eval.sh \
    --world-size 1 --rank 0 --workers 0 --epochs ${EPOCH} --arch ${ARCH} \
    --dist-url "tcp://datalab.cse.tamu.edu:${PORT}" \
    --valid_batch_size ${VALID_BATCH_SIZE} --train_batch_size ${TRAIN_BATCH_SIZE}