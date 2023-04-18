#!/bin/bash

TRACE_DIR="/home/pengmiao/Disk/work/data/Graph_dataset/IPDPS/GPOP/ChampSimTraces"
OUTPUT_DIR="/home/pengmiao/Disk/work/data/Graph_dataset/IPDPS/GPOP/Results"
PREFETCH_DIR="/home/pengmiao/Disk/work/data/Graph_dataset/IPDPS/GPOP/Prefetching"

PREFETCH_VERSION="Perfect_next_line"

BINARY=${1}
N_WARM=${2}
N_SIM=${3}
N_MIX=${4}
PHASE=${5}
APP=${6}
PREFETCH=${PREFETCH_DIR}/${PREFETCH_VERSION}/${APP}.${PHASE}.pref.txt

if [ $PHASE = "scatter" ]; then
    TRACE0=${APP}.s1.gz
    TRACE1=${APP}.s2.gz
    TRACE2=${APP}.s3.gz
    TRACE3=${APP}.s4.gz
fi
if [ $PHASE = "gather" ]; then
    TRACE0=${APP}.g1.gz
    TRACE1=${APP}.g2.gz
    TRACE2=${APP}.g3.gz
    TRACE3=${APP}.g4.gz
fi

(< ${PREFETCH} ../bin/${BINARY} -warmup_instructions ${N_WARM}000000 -simulation_instructions ${N_SIM}000000 -traces ${TRACE_DIR}/${TRACE0} ${TRACE_DIR}/${TRACE1} ${TRACE_DIR}/${TRACE2} ${TRACE_DIR}/${TRACE3}) &> ${OUTPUT_DIR}/${APP}.${PHASE}.${BINARY}.${PREFETCH_VERSION}.txt 2>&1

# ./run_4core_my.sh bimodal-no-no-no-no-lru-4core 0 10 0 scatter bfs.amazon.trace