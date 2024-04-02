#!/usr/bin/bash
set -ex
BIN=`dirname ${0}`
BIN=`cd ${BIN}; pwd`
ROOT=`cd ${BIN}; cd ../; pwd`

export ENABLE_EXPERIMENTAL_FLAGS=1
export VISUALIZATION_MODE=0
export GRAPH_VISUALIZATION=1
export HABANA_LOGS=`pwd`/logs
export LOG_LEVEL_ALL=1


export GC_KERNEL_PATH=/root/g2/g2attn/libcustom_tpc_perf_lib.so:${GC_KERNEL_PATH}

#python test_bmm.py
python test_attn.py

