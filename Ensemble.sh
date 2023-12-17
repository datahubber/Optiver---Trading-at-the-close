#!/bin/sh

TIME_HOME="/home/joseph/Projects/Optiver---Trading-at-the-close"
TF_HOME="/home/joseph/anaconda3/envs/kaggle/bin"
TIME_NOW=`date +"%Y-%m-%d_%H-%M-%S"`

MODEL_NAME=SE200

nohup ${TF_HOME}/python -u ${TIME_HOME}/Ensemble/${MODEL_NAME}.py  > ${TIME_HOME}/bash_logs/${MODEL_NAME}_${TIME_NOW}.log 2>&1 &


