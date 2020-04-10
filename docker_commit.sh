#!/usr/bin/env bash

CONTAINER_ID=$1

docker commit -a "yiwei_chen@tamu.edu" \
    -m "install torch and torchvision" \
    $CONTAINER_ID distributed_nas:latest
