#!/usr/bin/env bash

GPU=3,2
BATCH_SIZE=68
SEED=2
EPOCH=50
MICRO_BATCH_RATIO=0.5
EXP_PATH="exp/pipeline/nvtx_layers4_sep_alpha_blockForward_batch_size${BATCH_SIZE}_micro${MICRO_BATCH_RATIO}_gpu${GPU}"

nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx --output pipeline_nasp --delay=5 python \
    train_search.py --data /home/yiwei/cifar10 --batch_size ${BATCH_SIZE} --gpu ${GPU} \
    --save ${EXP_PATH} --seed ${SEED} --exec_script scripts/pipeline_search.sh --epochs ${EPOCH} \
    --worker 0 --micro_batch_ratio ${MICRO_BATCH_RATIO} --layers 4 --report_freq 1