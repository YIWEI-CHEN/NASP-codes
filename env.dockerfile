FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
LABEL maintainer="Yi-Wei Chen <yiwei_chen@tamu.edu>"

RUN apt-get update && apt-get install \
  -y --no-install-recommends python3 python3-pip

RUN pip3 install torch==1.3.0 torchvision==0.4.1