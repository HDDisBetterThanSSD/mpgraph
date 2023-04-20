from preprocessing import preprocessing_phase,read_load_trace_data
import numpy as np
import pandas as pd
import config as cf
from sklearn import tree
from sklearn.metrics import accuracy_score
from ks_test import KSWIN_soft, KSWIN_std_D, KSWIN_soft_D

BLOCK_BITS=cf.BLOCK_BITS
TOTAL_BITS=cf.TOTAL_BITS
LOOK_BACK=cf.LOOK_BACK
PRED_FORWARD=cf.PRED_FORWARD

BLOCK_NUM_BITS=cf.BLOCK_NUM_BITS
PAGE_BITS=cf.PAGE_BITS
BITMAP_SIZE=cf.BITMAP_SIZE
DELTA_BOUND=cf.DELTA_BOUND
SPLIT_BITS=cf.SPLIT_BITS
FILTER_SIZE=cf.FILTER_SIZE



TRAIN_NUM=1
TOTAL_NUM =4
SKIP_NUM = 0

TRANSITION_NUM=10

WINDOW_SIZE=480
STATE_SIZE = 120

THREAD_R=0.6
SOFT_WINDOW = WINDOW_SIZE+STATE_SIZE

#%%

def ks_detection_D(model_type, X):
    if model_type == "std":
        kswin =KSWIN_std_D(alpha=0.001, window_size=WINDOW_SIZE,  stat_size=STATE_SIZE, seed=1)
    elif model_type == "soft":
        kswin =KSWIN_soft_D(alpha=0.001, window_size=WINDOW_SIZE,  stat_size=STATE_SIZE, seed=1, th_r=THREAD_R)
    else:
        print("error in k-s test model type.")
        return
    
    st_list=[]
    detect=[]
    df=pd.DataFrame({})
    
    for i, val in enumerate(X):
        kswin.update(val)
        st_list.append(kswin.st)
        if kswin._drift_detected:
            detect.append(1)
        else:
            detect.append(0)
    
    df["st"]=st_list
    df["threshold"] = kswin.threshold
    df["detect"]=detect
    
    return df

def check_shift(y_eval,y_eval_shift):
    if y_eval==y_eval_shift:
        return 0
    else:
        return 1

#%%
def phase_detection_ks(file_path_s,file_path_g):
    train_data_s, eval_data_s = read_load_trace_data(file_path_s, TRAIN_NUM,TOTAL_NUM,SKIP_NUM)
    train_data_g, eval_data_g = read_load_trace_data(file_path_g, TRAIN_NUM,TOTAL_NUM,SKIP_NUM)
    
    
    df_eval_s=preprocessing_phase(eval_data_s,0)
    df_eval_g=preprocessing_phase(eval_data_g,1)
    
    df_list=[]
    for i in range(TRANSITION_NUM):
        df_list.append(df_eval_s)
        df_list.append(df_eval_g)
    #    df_mix_train=pd.concat([df_train_g, df_train_a,df_train_s], axis=0)
    #    df_mix_train = df_mix_train.sample(frac = 1) #shuffle
    
    df=pd.concat(df_list,axis=0)
    
    
    df["y_phase_shift"]=df['phase'].shift(1)
    df=df.dropna()
    df["shift"] = df.apply(lambda x: check_shift(x["phase"],x["y_phase_shift"]),axis=1)
    
    
    X_e=df.ip.values.tolist()
    
    df_std= ks_detection_D("std",X_e)
    df_soft= ks_detection_D("soft",X_e)
    
    print("kswin processing...")
    
    df["ks_detect"]=df_std["detect"].values.tolist()
    
    print("soft ks processing...")
    df["soft_ks_detect"]=df_soft["detect"].values.tolist()
    
    print("Standard KSWIN detection")
    print(df_std[df_std["detect"]==1])
    
    print("Soft-KSWIN detection")
    print(df_soft[df_soft["detect"]==1])

    return df

#%%

def precision_recall_f1(tp,fp,fn):
    if tp <0:
        tp=0
    if fp <0:
        fp=0
    if fn <0:
        fn=0
    
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    f1 = 2* (p*r)/(p+r)
    return p, r, f1


#%%
def get_metrics(df):
    #shift_index_list=df[df["shift"]==1].index.tolist()
    shift_index_list=np.where(df["shift"].values==1)[0]
    
    dt_list, soft_dt_list = df["ks_detect"].values, df["soft_ks_detect"].values
    
    dt_tp, soft_dt_tp = 0,0
    dt_p, soft_dt_p= dt_list.sum(), soft_dt_list.sum()

    for idx in shift_index_list:
        if dt_list[idx:idx+SOFT_WINDOW].sum()>0:
            dt_tp+=1
        if soft_dt_list[idx:idx+SOFT_WINDOW].sum()>0:
            soft_dt_tp+=1
    
    dt_fp, soft_dt_fp = dt_p-dt_tp, soft_dt_p - soft_dt_tp
    dt_fn, soft_dt_fn = len(shift_index_list) - dt_tp, len(shift_index_list) - soft_dt_tp
    
    
    p_dt, r_dt, f1_dt= precision_recall_f1(dt_tp,dt_fp,dt_fn)
    p_soft_dt, r_soft_dt, f1_soft_dt= precision_recall_f1(soft_dt_tp,soft_dt_fp,soft_dt_fn)
    
    
    #print(p_dt, r_dt, f1_dt)
    #print(p_soft_dt, r_soft_dt, f1_soft_dt)
    
    return p_dt, r_dt, f1_dt, p_soft_dt, r_soft_dt, f1_soft_dt
    

def run_ks_2_phase(file_path_s,file_path_g,res_output_path):

    df_res= phase_detection_ks(file_path_s,file_path_g)
    
    p_dt, r_dt, f1_dt, p_soft_dt, r_soft_dt, f1_soft_dt = get_metrics(df_res)
    
    df=pd.DataFrame({})
    
    df["p_dt"], df["r_dt"], df["f1_dt"], df["p_soft_dt"], df["r_soft_dt"], df["f1_soft_dt"], df["win_size"], df["stat_size"] = \
        [p_dt], [r_dt], [f1_dt], [p_soft_dt], [r_soft_dt], [f1_soft_dt],  [WINDOW_SIZE], [STATE_SIZE]
    pd.DataFrame(df).to_csv(res_output_path,header=1, index=False, sep=" ") 
    print ("Results")
    print ("Standard KSWIN:")
    print(df[["p_dt","r_dt","f1_dt"]])

    print ("Soft-KSWIN:")
    print(df[["p_soft_dt","r_soft_dt","f1_soft_dt"]])

#%%



if __name__ == '__main__':
    file_path_s="/home/pengmiao/Disk/work/data/Graph_dataset/IPDPS/Powergraph/LoadTraces/pagerank.amazon.trace.scatter.txt"
    file_path_g="/home/pengmiao/Disk/work/data/Graph_dataset/IPDPS/Powergraph/LoadTraces/pagerank.amazon.trace.gather.txt"
    #file_path_a="/home/pengmiao/Disk/work/data/Graph_dataset/IPDPS/Powergraph/LoadTraces/pagerank.amazon.trace.apply.txt"
    res_output_path = "./KS_test_output.csv"


    run_ks_2_phase(file_path_s,file_path_g,res_output_path)

