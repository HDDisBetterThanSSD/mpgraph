#!/usr/bin/env python3

import os
import sys
#from data_loader import data_generator
from preprocessing import read_load_trace_data, preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#alg_list=["pagerank","connected_component","sssp","undirected_triangle_count"]
#app_list=["amazon","google","rmat-1","roadCA","soclj","wiki","youtube"]

alg_list=["sssp","undirected_triangle_count"]
app_list=["youtube","amazon","roadCA"]

phase_list=["gather","scatter"]

input_path_root="/data/pengmiao/Graph_dataset/IPDPS/PowerGraph/LoadTraces/"

output_path_root="/data/pengmiao/Graph_dataset/IPDPS/PowerGraph/LoadTraces_pd/"

NUM=20


def data_generator(file_path,NUM,output_file,Read_Pickle=False,only_val=False):
    
    _,data=read_load_trace_data(file_path[0],0,NUM)
    
    df_s = preprocessing(data)
    
    df_g = preprocessing(read_load_trace_data(file_path[1],0,NUM)[1])

    
    df_train_s, df_test_s = train_test_split(df_s, test_size=0.5)
    df_train_g, df_test_g = train_test_split(df_g, test_size=0.5)


    df_train_s.to_pickle(output_file+'.scatter'+".train.pkl")
    print ("output to pickle: ", output_file+'.scatter'+".train.pkl")
    df_test_s.to_pickle(output_file+'.scatter'+".test.pkl")
    print ("output to pickle: ", output_file+'.scatter'+".test.pkl")
    
    df_train_g.to_pickle(output_file+'.gather'+".train.pkl")
    print ("output to pickle: ", output_file+'.gather'+".train.pkl")
    df_test_g.to_pickle(output_file+'.gather'+".test.pkl")
    print ("output to pickle: ", output_file+'.gather'+".test.pkl")

    return


def main():
    os.makedirs(output_path_root, exist_ok=True)
    for alg in alg_list:
        for app in app_list:
            input_file_list=[]
            output_file=output_path_root+alg+"."+app
            for phase in phase_list:
                
                input_file=input_path_root+alg+"."+app+".trace."+phase+".txt"
                print(input_file)
                if os.path.exists(input_file):
                    input_file_list.append(input_file)
                else:
                    print("not exist file:",input_file)
                    pass
            if len(input_file_list) == len(phase_list):
                data_generator(input_file_list, NUM, output_file,Read_Pickle=False)
                
                

if __name__ == '__main__':
    main()