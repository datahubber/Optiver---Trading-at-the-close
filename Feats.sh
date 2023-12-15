#!/bin/sh

TIME_HOME="/home/joseph/Projects/Optiver---Trading-at-the-close"
TF_HOME="/home/joseph/anaconda3/envs/kaggle/bin"
TIME_NOW=`date +"%Y-%m-%d_%H-%M-%S"`

MODEL_NAME=Feats184

#${TF_HOME}/python -u ${TIME_HOME}/LGB/LGB.py

#nohup ${TF_HOME}/python -u ${TIME_HOME}/LGB/LGB.py  > /dev/null 2>&1 &
#nohup ${TF_HOME}/python -u ${TIME_HOME}/Feats/Feats.py  > ${TIME_HOME}/bash_logs/Feats_${TIME_NOW}.log 2>&1 &
#nohup ${TF_HOME}/python -u ${TIME_HOME}/Feats161/Feats161.py  > Feats161.log 2>&1 &

nohup ${TF_HOME}/python -u ${TIME_HOME}/Feats/${MODEL_NAME}.py  > ${TIME_HOME}/bash_logs/${MODEL_NAME}_${TIME_NOW}.log 2>&1 &


