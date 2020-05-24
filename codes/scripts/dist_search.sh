#!/usr/bin/env bash

GPU="2,3"
BATCH_SIZE=136
SEED=2
PORT=50019
EPOCH=50
EXP_PATH="exp/distributed/batch_size${BATCH_SIZE}_gpu${GPU}"

python train_search.py --data /home/yiwei/cifar10 --batch_size ${BATCH_SIZE} --gpu ${GPU} \
    --save ${EXP_PATH} --seed ${SEED} --exec_script scripts/dist_search.sh \
    --world-size 1 --rank 0 --workers 0 --epochs ${EPOCH} \
    --dist-url "tcp://datalab.cse.tamu.edu:${PORT}" \
    --report_freq 50