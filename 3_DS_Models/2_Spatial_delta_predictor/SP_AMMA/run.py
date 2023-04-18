#!/usr/bin/env python3

import os
import sys
from train import train_main

def main():
    alg_list=[sys.argv[1]]
    app_list=[sys.argv[2]]
    NUM=int(sys.argv[3])

    PHASE=sys.argv[4]
    #"scatter"

    input_path_root="../../../data/GPOP/LoadTraces_pd/"
    output_path_root="../../../res/2_Spatial_delta_predictor/"

    os.makedirs(output_path_root, exist_ok=True)
    for alg in alg_list:
        for app in app_list:
            input_file=input_path_root+alg+"."+app
            output_file=output_path_root+alg+"."+app    

            print("input_file:",input_file)
            print("output_file:",output_file)
            
            if os.path.exists(input_file+"."+PHASE+".train.pkl"):
                train_main(input_file,output_file,NUM,PHASE)
            else:
                print("not exist file:",input_file)
                pass
                

if __name__ == '__main__':
    main()