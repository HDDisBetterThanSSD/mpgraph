#!/usr/bin/env python3

import os
import sys
from phase_detection import run_dt_2_phase



def main():
    alg=sys.argv[1]#bfs
    app=sys.argv[2]#amazon
    alg_list, app_list, phase_list =[alg],[app],["scatter","gather"]
    input_path_root="../../../data/GPOP/LoadTraces/"
    output_path_root="../../../res/1_Phase_detection/2_soft_DT/"
    
    os.makedirs(output_path_root, exist_ok=True)
    for alg in alg_list:
        for app in app_list:
            input_file_list=[]
            output_file=output_path_root+alg+"."+app+".res"+".txt"       
            for phase in phase_list:
                input_file=input_path_root+alg+"."+app+".trace."+phase+".txt"
                if os.path.exists(input_file):
                    input_file_list.append(input_file)
                else:
                    print("not exist file:",input_file)
                    pass
            if len(input_file_list) == len(phase_list):
                run_dt_2_phase(input_file_list[0],input_file_list[1],output_file)


if __name__ == '__main__':
    main()