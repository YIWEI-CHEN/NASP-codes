#!/usr/bin/env bash

GPU=2
TRAIN_BATCH_SIZE=96
VALID_BATCH_SIZE=96
SEED=0
EPOCH=600
ARCH="SINGLE_NASP_0701"
PORTION=0.9
EXP_PATH="exp/single/${ARCH}/${PORTION}train_bz${TRAIN_BATCH_SIZE}_valid_bz${VALID_BATCH_SIZE}_gpu${GPU}"

python train.py --data /home/yiwei/cifar10 --gpu ${GPU} --save ${EXP_PATH} \
    --seed ${SEED} --epochs ${EPOCH} --arch ${ARCH} --workers 0 --cutout --auxiliary \
    --valid_batch_size ${VALID_BATCH_SIZE} --train_batch_size ${TRAIN_BATCH_SIZE} \
    --train_portion ${PORTION}
