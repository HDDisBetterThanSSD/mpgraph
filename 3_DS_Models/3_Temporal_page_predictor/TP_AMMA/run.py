#!/usr/bin/env python3

import os
import sys
from train import train_main

def main():
    alg_list=[sys.argv[1]]
    app_list=[sys.argv[2]]
    NUM=int(sys.argv[3])
    PHASE=sys.argv[4]

#    alg_list=["cc"]
#    app_list=["soclj"]
#    NUM=int(1)
#    PHASE="scatter"

    input_path_root="../../../data/GPOP/LoadTraces/"
    output_path_root="../../../res/3_Temporal_page_predictor/"



    os.makedirs(output_path_root, exist_ok=True)
    for alg in alg_list:
        for app in app_list:
            input_file_1=input_path_root+alg+"."+app+".trace."+PHASE+".txt"
            
            #input_file_2=input_path_root+alg+"."+app+".trace.gather.txt"
            
            output_file=output_path_root+alg+"."+app


            print("input_file:",input_file_1)
            print("output_file:",output_file)
            
            if os.path.exists(input_file_1):
                #phase="scatter"
                train_main(input_file_1,input_file_1,output_file,TOTAL_NUM=NUM,PHASE=PHASE)
                #phase ="gather"
                #train_main(input_file_2,input_file_2,output_file,TOTAL_NUM=NUM,PHASE=phase)
                #phase ="mix"
                #train_main(input_file_1,input_file_2,output_file,TOTAL_NUM=NUM,PHASE=phase)

            else:
                print("not exist file:",input_file_1)
                pass

    
if __name__ == '__main__':
    main()
