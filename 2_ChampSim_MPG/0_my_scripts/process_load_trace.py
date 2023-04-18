#!/usr/bin/env python3

import os
import sys

#app_list=["google","roadCA","soclj","wiki","youtube"]
#alg_list=["bfs","cc","nibble","pr","sssp"]
#alg_list=["bfs"]
#app_list=["amazon"]
#app_list=["amazon","google","roadCA","soclj","wiki","youtube"]
#phase_list=["gather"]

#def read_file(input_path,output_path):

#input_path_root="/home/pengmiao/Disk/work/data/Graph_dataset/IPDPS/GPOP/LoadTraces_raw/"
#output_path_root="/home/pengmiao/Disk/work/data/Graph_dataset/IPDPS/GPOP/LoadTraces/"

#input_path_root="/home/pengmiao/Disk/work/data/Graph_dataset/IPDPS/xstream/LoadTraces_raw/"
#output_path_root="/home/pengmiao/Disk/work/data/Graph_dataset/IPDPS/xstream/LoadTraces/"



#%%

def filter_file(input_file,output_file):
    with open(input_file, 'r') as f_in:
        with open(output_file, "w") as f_out:
            for line in f_in:
                if any(map(str.isupper, line)):
                    pass
                elif line == "\n":
                    pass
                else:
                    f_out.write(line)
                
#%%

               


if __name__ == '__main__':
    alg=sys.argv[1]
    app=sys.argv[2]
    alg_list, app_list, phase_list =[alg],[app],["scatter","gather"]
#    input_path_root="/home/pengmiao/Disk/Lab/SC23/Draft/data/GPOP/LoadTraces_raw/"
#    output_path_root="/home/pengmiao/Disk/Lab/SC23/Draft/data/GPOP/LoadTraces/"

    input_path_root="../../data/GPOP/LoadTraces_raw/"
    output_path_root="../../data/GPOP/LoadTraces/"

    if not os.path.exists(output_path_root):
        os.makedirs(output_path_root)

    for alg in alg_list:
        for app in app_list:
            for phase in phase_list:
                input_file=input_path_root+alg+"."+app+".trace."+phase+".txt"
                output_file=output_path_root+alg+"."+app+".trace."+phase+".txt"
                if os.path.exists(input_file):
                    filter_file(input_file,output_file)
                    print("Processed trace stored in:",output_file)
                else:
                    print("not exist file:",input_file)