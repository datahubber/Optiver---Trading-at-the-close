#!/bin/sh

TIME_HOME="/home/joseph/Projects/Optiver---Trading-at-the-close"
TF_HOME="/home/joseph/anaconda3/envs/kaggle/bin"


nohup ${TF_HOME}/python -u ${TIME_HOME}/LGB.py  > /dev/null 2>&1 &


