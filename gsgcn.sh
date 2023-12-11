#!/bin/sh

TIME_HOME="/home/joseph/Projects/graph_time_forecast"
TF_HOME="/home/joseph/anaconda3/envs/tensorflow/bin"


#${TF_HOME}/python -u ${TIME_HOME}/GSGCN/gsgcn.py -i 30 -b 64 -s ${STEP} -g 0 -e 100 -d 'alabama' -p 'adj_mat.pkl' -n 5

DATASET='alabama'
PKL='adj_mat.pkl'
NODES=5

# alabama
#nohup ${TF_HOME}/python -u ${TIME_HOME}/GSGCN/gsgcn2.py -i 40 -b 128 -s 5 -g 0 -e 100 -d ${DATASET} -p ${PKL} -n ${NODES} > /dev/null 2>&1 &
#sleep 2
#nohup ${TF_HOME}/python -u ${TIME_HOME}/GSGCN/gsgcn2.py -i 40 -b 128 -s 10 -g 1 -e 100 -d ${DATASET} -p ${PKL} -n ${NODES} > /dev/null 2>&1 &
#sleep 2
#nohup ${TF_HOME}/python -u ${TIME_HOME}/GSGCN/gsgcn2.py -i 40 -b 128 -s 15 -g 0 -e 100 -d ${DATASET} -p ${PKL} -n ${NODES} > /dev/null 2>&1 &
#sleep 2


#DATASET = 'jjj5_do'
#PKL = 'jjj5_adj_mat.pkl'
#NODES = 5

# jjj5_do
#nohup ${TF_HOME}/python -u ${TIME_HOME}/GSGCN/gsgcn.py -i 40 -b 64 -s 5 -g 0 -e 60 -d ${DATASET} -p ${PKL} -n ${NODES} > /dev/null 2>&1 &
#sleep 2
#nohup ${TF_HOME}/python -u ${TIME_HOME}/GSGCN/gsgcn.py -i 40 -b 64 -s 10 -g 1 -e 60 -d ${DATASET} -p ${PKL} -n ${NODES} > /dev/null 2>&1 &
#sleep 2
#nohup ${TF_HOME}/python -u ${TIME_HOME}/GSGCN/gsgcn.py -i 40 -b 64 -s 15 -g 2 -e 60 -d ${DATASET} -p ${PKL} -n ${NODES} > /dev/null 2>&1 &
#sleep 2

DATASET='jjj6_do'
PKL='jjj6_adj_mat.pkl'
NODES=6

nohup ${TF_HOME}/python -u ${TIME_HOME}/GSGCN/gsgcn2.py -i 40 -b 32 -s 5 -g 0 -e 100 -d ${DATASET} -p ${PKL} -n ${NODES} > /dev/null 2>&1 &
sleep 2
nohup ${TF_HOME}/python -u ${TIME_HOME}/GSGCN/gsgcn2.py -i 40 -b 64 -s 10 -g 1 -e 100 -d ${DATASET} -p ${PKL} -n ${NODES} > /dev/null 2>&1 &
sleep 2
nohup ${TF_HOME}/python -u ${TIME_HOME}/GSGCN/gsgcn2.py -i 40 -b 64 -s 15 -g 1 -e 100 -d ${DATASET} -p ${PKL} -n ${NODES} > /dev/null 2>&1 &
sleep 2


DATASET='alabama'
PKL='adj_mat.pkl'
NODES=5

DATASET='jjj6_do'
PKL='jjj6_adj_mat.pkl'
NODES=6
STEP=5


# n_in
#echo "gsgcn"
#nohup ${TF_HOME}/python -u ${TIME_HOME}/GSGCN/gsgcn.py -i 60 -b 64 -s ${STEP} -g 0 -e 100 -d ${DATASET} -p ${PKL} -n ${NODES} > /dev/null 2>&1 &
#sleep 2
#nohup ${TF_HOME}/python -u ${TIME_HOME}/GSGCN/gsgcn.py -i 50 -b 64 -s ${STEP} -g 0 -e 100 -d ${DATASET} -p ${PKL} -n ${NODES} > /dev/null 2>&1 &
#sleep 2
#nohup ${TF_HOME}/python -u ${TIME_HOME}/GSGCN/gsgcn.py -i 40 -b 64 -s ${STEP} -g 1 -e 100 -d ${DATASET} -p ${PKL} -n ${NODES} > /dev/null 2>&1 &
#sleep 2
#nohup ${TF_HOME}/python -u ${TIME_HOME}/GSGCN/gsgcn.py -i 30 -b 64 -s ${STEP} -g 1 -e 100 -d ${DATASET} -p ${PKL} -n ${NODES} > /dev/null 2>&1 &
#sleep 2
#nohup ${TF_HOME}/python -u ${TIME_HOME}/GSGCN/gsgcn.py -i 20 -b 64 -s ${STEP} -g 1 -e 100 -d ${DATASET} -p ${PKL} -n ${NODES} > /dev/null 2>&1 &
#sleep 2

# hidden_size
#echo "gsgcn"
#nohup ${TF_HOME}/python -u ${TIME_HOME}/GSGCN/gsgcn.py -i 40 -b 64 -s ${STEP} -r 32 -g 0 -e 100 -d ${DATASET} -p ${PKL} -n ${NODES} > /dev/null 2>&1 &
#sleep 2
#nohup ${TF_HOME}/python -u ${TIME_HOME}/GSGCN/gsgcn.py -i 40 -b 64 -s ${STEP} -r 64 -g 0 -e 100 -d ${DATASET} -p ${PKL} -n ${NODES} > /dev/null 2>&1 &
#sleep 2
#nohup ${TF_HOME}/python -u ${TIME_HOME}/GSGCN/gsgcn.py -i 40 -b 64 -s ${STEP} -r 128 -g 1 -e 100 -d ${DATASET} -p ${PKL} -n ${NODES} > /dev/null 2>&1 &
#sleep 2
#nohup ${TF_HOME}/python -u ${TIME_HOME}/GSGCN/gsgcn.py -i 40 -b 64 -s ${STEP} -r 16 -g 1 -e 100 -d ${DATASET} -p ${PKL} -n ${NODES} > /dev/null 2>&1 &
#sleep 2
#nohup ${TF_HOME}/python -u ${TIME_HOME}/GSGCN/gsgcn.py -i 40 -b 64 -s ${STEP} -r 256 -g 2 -e 100 -d ${DATASET} -p ${PKL} -n ${NODES} > /dev/null 2>&1 &
#sleep 2

HIDDEN=32
# gsgcn_num
#echo "gsgcn gsgcn_num"
#nohup ${TF_HOME}/python -u ${TIME_HOME}/GSGCN/gsgcn.py -i 40 -b 64 -s ${STEP} -c 1 -r ${HIDDEN} -g 0 -e 100 -d ${DATASET} -p ${PKL} -n ${NODES} > /dev/null 2>&1 &
#sleep 2
#nohup ${TF_HOME}/python -u ${TIME_HOME}/GSGCN/gsgcn.py -i 40 -b 64 -s ${STEP} -c 2 -r ${HIDDEN} -g 2 -e 100 -d ${DATASET} -p ${PKL} -n ${NODES} > /dev/null 2>&1 &
#sleep 2
#nohup ${TF_HOME}/python -u ${TIME_HOME}/GSGCN/gsgcn.py -i 40 -b 64 -s ${STEP} -c 3 -r ${HIDDEN} -g 1 -e 100 -d ${DATASET} -p ${PKL} -n ${NODES} > /dev/null 2>&1 &
#sleep 2
#nohup ${TF_HOME}/python -u ${TIME_HOME}/GSGCN/gsgcn.py -i 40 -b 64 -s ${STEP} -c 4 -r ${HIDDEN} -g 1 -e 100 -d ${DATASET} -p ${PKL} -n ${NODES} > /dev/null 2>&1 &
#sleep 2

HIDDEN=16
GSGCN=3
EPOCH=100
#nohup ${TF_HOME}/python -u ${TIME_HOME}/GSGCN/gsgcn.py -i 40 -b 64 -s ${STEP} -c ${GSGCN} -r ${HIDDEN} -g 0 -e ${EPOCH} -d ${DATASET} -p ${PKL} -n ${NODES} > /dev/null 2>&1 &
#sleep 2
#nohup ${TF_HOME}/python -u ${TIME_HOME}/GSGCN/gsgcn.py -i 40 -b 64 -s ${STEP} -c ${GSGCN} -r ${HIDDEN} -g 2 -e ${EPOCH} -d ${DATASET} -p ${PKL} -n ${NODES} > /dev/null 2>&1 &
#sleep 2
#nohup ${TF_HOME}/python -u ${TIME_HOME}/GSGCN/gsgcn.py -i 40 -b 64 -s ${STEP} -c ${GSGCN} -r ${HIDDEN} -g 1 -e ${EPOCH} -d ${DATASET} -p ${PKL} -n ${NODES} > /dev/null 2>&1 &
#sleep 2


unset TIME_HOME
unset TF_HOME

