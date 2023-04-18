#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 21:42:25 2022

@author: pengmiao
"""
import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd

NEXT_LINE_DISTANCE=1
DEGREE=2
# =============================================================================
# alg_list=["bfs","cc","nibble","pr","sssp"]
# app_list=["amazon","google","roadCA","soclj","wiki","youtube"]
# phase_list=["scatter","gather"]
# =============================================================================

#alg_list=["bfs","cc","nibble","pr","sssp"]
alg_list=["pr"]
app_list=["amazon"]
phase_list=["scatter","gather"]

#def read_file(input_path,output_path):

input_path_root="/home/pengmiao/Disk/work/data/Graph_dataset/IPDPS/GPOP/LoadTraces/"
output_path_root="/home/pengmiao/Disk/work/data/Graph_dataset/IPDPS/GPOP/Prefetching/Perfect_next_line/"

# =============================================================================
# alg=alg_list[0]
# app=app_list[0]
# phase=phase_list[0]
# input_file=input_path_root+alg+"."+app+".trace."+phase+".txt"
# output_file=output_path_root+alg+"."+app+".trace."+phase+".pref.txt"
# =============================================================================
#%%

def perfect_next_line(input_file,output_file):
    trace_pd=pd.read_csv(input_file,sep=", ")
    trace_pd.columns=["id", "cycle", "addr", "ip", "hit"]  
    trace_pd["addr_1"]=trace_pd["addr"].shift(0-NEXT_LINE_DISTANCE)
    trace_pd2=trace_pd[["id","addr_1"]].dropna()
    trace_pd2.to_csv(output_file, header=None, index=False, sep=" ")
    
def perfect_next_line_2(input_file,output_file):
    trace_pd=pd.read_csv(input_file,sep=", ")
    trace_pd.columns=["id", "cycle", "addr", "ip", "hit"]  
    trace_pd["addr_1"]=trace_pd["addr"].shift(0-NEXT_LINE_DISTANCE)
    trace_pd["addr_2"]=trace_pd["addr"].shift(0-NEXT_LINE_DISTANCE-1)
    trace_pd2=trace_pd[["id","addr_1","addr_2"]].dropna()
    
    trace_pd2['combined']= trace_pd2[["addr_1","addr_2"]].values.tolist()
    trace_pd2=trace_pd2.explode("combined")
    trace_pd2[["id","combined"]].to_csv(output_file, header=None, index=False, sep=" ")

def main():
    for alg in alg_list:
        for app in app_list:
            for phase in phase_list:
                input_file=input_path_root+alg+"."+app+".trace."+phase+".txt"
                output_file=output_path_root+alg+"."+app+".trace."+phase+".pref.txt"
                if os.path.exists(input_file):
                    perfect_next_line(input_file,output_file)
                    #perfect_next_line_2(input_file,output_file)
                else:
                    print("not exist file:",input_file)

if __name__ == '__main__':
    main()

