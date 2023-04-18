#!/bin/bash


mkdir -p "../../data/csr"

Root_csr_code="../pcpm/csr_gen"

cd $Root_csr_code

Graph_data_root="../../../data/raw"
Output_path_root="../../../data/csr"

make

DEFAULT_APP="amazon0302.txt"
DEFAULT_OUT="amazon"


app1=${1:-$DEFAULT_APP}
output=${2:-$DEFAULT_OUT}


echo $app1
#Graph_data_input="/data/pengmiao/Graph_dataset/raw/""$1"
Graph_data_input=${Graph_data_root}/$app1
#Graph_csr_output="/data/pengmiao/Graph_dataset/csr_w/""$2"
Graph_csr_output=${Output_path_root}/$output
./a.out $Graph_data_input $Graph_csr_output
echo $Graph_csr_output' csr generated'


#./2_gen_gpop_csr.sh com-amazon.ungraph.txt amazon
