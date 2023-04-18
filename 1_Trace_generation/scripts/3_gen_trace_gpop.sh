#!/bin/bash

DEFAULT_ALG="bfs"
DEFAULT_APP="amazon"
DEFAULT_LENGTH=10000000

algorithm=${1:-$DEFAULT_ALG}
app=${2:-$DEFAULT_APP}
app_list=($app)
Length_Instr=${3:-$DEFAULT_LENGTH}

mkdir -p "../../data/GPOP"
mkdir -p "../../data/GPOP/ChampSimTraces"


ROOT="$PWD"

Root_Algorithm=$ROOT"/../GPOP_labeled/"$algorithm
cd $Root_Algorithm

rm $algorithm
rm ${algorithm}.o

make

Algorithm_exe=$ROOT"/../GPOP_labeled/$algorithm/$algorithm"

Root_ChampSim_tracer="../../pintool_tracer"

cd $Root_ChampSim_tracer

./make_tracer.sh

Graph_data_root="../../data/csr"
Output_path_root="../../data/GPOP/ChampSimTraces"


Skip_Instr=0

Thread=4
Iteration=10

echo $Thread $Iteration

for app1 in ${app_list[*]}; do
	echo $app1
    Graph_data=$Graph_data_root/${app1}
    output_path=$Output_path_root/$algorithm.${app1}.trace

    #$Algorithm_exe $Graph_data -t $Thread -r $Iteration
    /root/pin-3.17-98314-g0c048d619-gcc-linux/pin -t obj-intel64/champsim_tracer.so -s $Skip_Instr -t $Length_Instr -o $output_path -- $Algorithm_exe $Graph_data -t $Thread -r $Iteration

    echo "done for app "$app1
done

#./3_gen_trace_gpop.sh cc amazon
