#!/usr/bin/env bash

# if [ ! -d "log" ]; then
#   mkdir log
# fi
# export PYTHONIOENCODING=UTF-8
# filename="log/0033_alldatafinetune`date +20%y_%m_%d___%H_%M_%S`.txt"
# CUDA_VISIBLE_DEVICES=1 python -u dzj_ctcbaseline.py \
# 	2>&1 | tee $filename


CUDA_VISIBLE_DEVICES=1 python dzj_ctcbaseline.py
