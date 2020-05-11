#!/usr/bin/env bash

GPU=2
BATCH_SIZE=68
SEED=2
EXP_PATH="exp/single/nvtx_layers4_batch_size${BATCH_SIZE}_gpu${GPU}"
EPOCH=50

nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx --output single_nasp --delay=5 python \
    train_search.py --data /home/yiwei/cifar10 --batch_size ${BATCH_SIZE} --gpu ${GPU} \
    --save ${EXP_PATH} --seed ${SEED} --exec_script scripts/search.sh --epochs ${EPOCH} \
    --worker 0 --layers 4 --report_freq 1