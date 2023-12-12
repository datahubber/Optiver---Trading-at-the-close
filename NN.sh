#!/bin/sh

TIME_HOME="/home/joseph/Projects/Optiver---Trading-at-the-close"
TF_HOME="/home/joseph/anaconda3/envs/kaggle/bin"
TIME_NOW=`date +"%Y-%m-%d_%H-%M-%S"`

#${TF_HOME}/python -u ${TIME_HOME}/LGB/LGB.py
#${TF_HOME}/python -u ${TIME_HOME}/NN/NN161.py

nohup ${TF_HOME}/python -u ${TIME_HOME}/NN/NN161.py  > ${TIME_HOME}/bash_logs/NN161_${TIME_NOW}.log 2>&1 &


