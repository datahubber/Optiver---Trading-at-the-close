#!/bin/sh

TIME_HOME="/home/joseph/Projects/Optiver---Trading-at-the-close"
TF_HOME="/home/joseph/anaconda3/envs/kaggle/bin"

#${TF_HOME}/python -u ${TIME_HOME}/LGB/LGB.py
#${TF_HOME}/python -u ${TIME_HOME}/NN/NN161.py

nohup ${TF_HOME}/python -u ${TIME_HOME}/NN/NN161.py  > /dev/null 2>&1 &


