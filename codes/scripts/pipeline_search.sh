#!/usr/bin/env bash

GPU=2,3
BATCH_SIZE=760
SEED=2
EPOCH=50
MICRO_BATCH_RATIO=0.5
CHUNKS=8
BALANCE="1:1"
EXP_PATH="exp/pipeline/gpipe_sep_alpha_infer_no_grad_batch_size${BATCH_SIZE}_chunk${CHUNKS}_balance${BALANCE}_gpu${GPU}"

#nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx --output pipeline_nasp --delay=5 python \
#    train_search.py --data /home/yiwei/cifar10 --batch_size ${BATCH_SIZE} --gpu ${GPU} \
#    --save ${EXP_PATH} --seed ${SEED} --exec_script scripts/pipeline_search.sh --epochs ${EPOCH} \
#    --worker 0 --micro_batch_ratio ${MICRO_BATCH_RATIO} --layers 4 --report_freq 1

python train_search.py --data /home/yiwei/cifar10 --batch_size ${BATCH_SIZE} --gpu ${GPU} \
    --save ${EXP_PATH} --seed ${SEED} --exec_script scripts/pipeline_search.sh --epochs ${EPOCH} \
    --worker 0 --chunks ${CHUNKS} --report_freq 10
