#!/usr/bin/env bash

GPU="5,6,7"
TRAIN_BATCH_SIZE=444
VALID_BATCH_SIZE=64
SEED=0
PORT=50017
EPOCH=600
ARCH="NASP"
PORTION=1.0
EXP_PATH="exp/distributed/${ARCH}/${PORTION}train_bz${TRAIN_BATCH_SIZE}_valid_bz${VALID_BATCH_SIZE}_gpu${GPU}"

python dist_train.py --data /home/yiwei/cifar10 --gpu ${GPU} --save ${EXP_PATH} \
    --seed ${SEED} --cutout --auxiliary --exec_script scripts/eval.sh \
    --world-size 1 --rank 0 --workers 0 --epochs ${EPOCH} --arch ${ARCH} \
    --dist-url "tcp://datalab3.engr.tamu.edu:${PORT}" \
    --valid_batch_size ${VALID_BATCH_SIZE} --train_batch_size ${TRAIN_BATCH_SIZE} \
    --train_portion ${PORTION} --exec_script scripts/dist_eval.sh
