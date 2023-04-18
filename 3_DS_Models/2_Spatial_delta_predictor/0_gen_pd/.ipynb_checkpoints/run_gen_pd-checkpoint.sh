#!/bin/bash

#LoadTrace_ROOT="/data/pengmiao/Graph_dataset/pr_block"
LoadTrace_ROOT="/data/pengmiao/Graph_dataset/GPOP/trace_2"
Python_ROOT="/home/pengmiao/Project/2022/GPOP/prediction/1_1_trans_basic/0_gen_pd"

#NUM=10000
NUM='-1'

algorithm_list=('bfs' 'cc' 'nibble' 'pr' 'sssp')

#app_list=(amazon google roadCA soclj wiki youtube)
app_list=(soclj wiki youtube)

for algorithm in ${algorithm_list[*]}; do
    for app1 in ${app_list[*]}; do
        echo $algorithm
        echo $app1
        echo "NUM: "$NUM
        file_path=$LoadTrace_ROOT/$algorithm.${app1}.trace.2
        python $Python_ROOT/train.py $file_path $NUM

        echo "Done for: "$file_path
    done
done

