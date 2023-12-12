#!/bin/sh

TIME_HOME="/home/joseph/Projects/Optiver---Trading-at-the-close"
TF_HOME="/home/joseph/anaconda3/envs/kaggle/bin"

#${TF_HOME}/python -u ${TIME_HOME}/LGB/LGB.py

#nohup ${TF_HOME}/python -u ${TIME_HOME}/LGB/LGB.py  > /dev/null 2>&1 &
#nohup ${TF_HOME}/python -u ${TIME_HOME}/Feats161/Feats161.py  > /dev/null 2>&1 &
nohup ${TF_HOME}/python -u ${TIME_HOME}/Feats161/Feats161.py  > Feats161.log 2>&1 &


