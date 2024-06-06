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


G2ATTN_DIR=`python -c "import g2attn;print(g2attn.__file__)"`
G2ATTN_DIR=`echo ${G2ATTN_DIR} | grep 'g2attn/__init__.py'`
G2ATTN_DIR=`dirname ${G2ATTN_DIR}`
G2ATTN_SO_FILE=${G2ATTN_DIR}/libcustom_tpc_perf_lib.so
export GC_KERNEL_PATH=${G2ATTN_SO_FILE}:${GC_KERNEL_PATH}

#python test_bmm.py
#python test_attn.py
#python profile_add_sum_idx_sv.py
python profile_add_sum_idx_qk.py
