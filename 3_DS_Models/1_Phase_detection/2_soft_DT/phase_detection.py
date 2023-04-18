from preprocessing import preprocessing_phase,read_load_trace_data
import numpy as np
import pandas as pd
import config as cf
from sklearn import tree
from sklearn.metrics import accuracy_score


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

SOFT_WINDOW = 50

SOFT_RATE=0.7

#%%

def phase_detection_decision_tree(file_path_s,file_path_g):
    train_data_s, eval_data_s = read_load_trace_data(file_path_s, TRAIN_NUM,TOTAL_NUM,SKIP_NUM)
    train_data_g, eval_data_g = read_load_trace_data(file_path_g, TRAIN_NUM,TOTAL_NUM,SKIP_NUM)
    
    df_train_s=preprocessing_phase(train_data_s,0)
    df_eval_s=preprocessing_phase(eval_data_s,0)
    
    df_train_g=preprocessing_phase(train_data_g,1)
    df_eval_g=preprocessing_phase(eval_data_g,1)
    
    
    df_list=[]
    for i in range(TRANSITION_NUM):
        df_list.append(df_eval_s)
        df_list.append(df_eval_g)

    df_mix_train=pd.concat([df_train_s,df_train_g], axis=0)
    df_mix_train = df_mix_train.sample(frac = 1) #shuffle
    
    df_mix_eval=pd.concat(df_list,axis=0)
    
    
    X=df_mix_train.past_ip.values.tolist()
    Y=df_mix_train.phase.values.tolist()
    
    X_e=df_mix_eval.past_ip.values.tolist()
    Y_e=df_mix_eval.phase.values.tolist()
    
    print("DT Training")
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)
    
    print("DT Predicting")
    Y_pred_eval=clf.predict(X_e)
    
    print("DT size:",clf.tree_.node_count)
    print("DT depth:", clf.get_depth())
    
    return Y_e,list(Y_pred_eval),clf


#%%
def check_shift(y_eval,y_eval_shift):
    if y_eval==y_eval_shift:
        return 0
    else:
        return 1

def reduce_soft_detection(y_pred_list):
    len_y=len(y_pred_list)
    len_y_a=len_y//2
    
    lst_a=y_pred_list[:len_y_a]
    lst_b=y_pred_list[len_y_a:]
    
    common_a=max(set(lst_a), key=lst_a.count)
    common_b=max(set(lst_b), key=lst_b.count)
    
    if common_a == common_b:
        return 0
    else:
        
        rate_a= lst_a.count(common_a)/len_y_a
        rate_b=lst_b.count(common_b)/(len_y-len_y_a)
        
        if rate_a>SOFT_RATE and rate_b>SOFT_RATE:
            return 1
        else:
            return 0
    

def check_soft_shift(a,b):
    if a<b:
        return 1
    else:
        return 0

#%%

def result_df(y_eval,y_pred_eval):
    
    df=pd.DataFrame({})
    df["y_eval"]=y_eval
    df["y_pred"]=y_pred_eval
    
    df["y_eval_shift"]=df['y_eval'].shift(1)
    df=df.dropna()
    
    df["shift"] = df.apply(lambda x: check_shift(x["y_eval"],x["y_eval_shift"]),axis=1)
    
    for i in range(SOFT_WINDOW):
            df['y_pred_shift_%d'%(i+1)]=df['y_pred'].shift(periods=(i+1))
    
    
    df["dt_detect"]=df.apply(lambda x: check_shift(x["y_pred"],x["y_pred_shift_1"]),axis=1)
    
    past_name=['y_pred_shift_%d'%(i) for i in range(SOFT_WINDOW,0,-1)]
    df["y_pred_shift_all"]=df[past_name].values.tolist()

    print("Soft-DT Predicting")
    
    df["soft_dt_detect_reduce"] = df.apply(lambda x: reduce_soft_detection(x["y_pred_shift_all"]),axis=1)
    
    df["soft_dt_detect_reduce_shift"] = df["soft_dt_detect_reduce"].shift(1)

    df=df.dropna()
    
    df["soft_dt_detect"] = df.apply(lambda x: check_soft_shift(x["soft_dt_detect_reduce"],x["soft_dt_detect_reduce_shift"]),axis=1)
    
    print(df[df["soft_dt_detect"]==1])
    
    return df[["y_eval","y_pred","shift","dt_detect","soft_dt_detect"]]
    


#%%
#TP,TN,FP,FN

def precision_recall_f1(tp,fp,fn):
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    f1 = 2* (p*r)/(p+r)
    return p, r, f1


#%%
def get_metrics(df):
    #shift_index_list=df[df["shift"]==1].index.tolist()
    shift_index_list=np.where(df["shift"].values==1)[0]
    
    dt_list, soft_dt_list = df["dt_detect"].values, df["soft_dt_detect"].values
    
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

    print(p_dt, r_dt, f1_dt)
    print(p_soft_dt, r_soft_dt, f1_soft_dt)
    
    
    return p_dt, r_dt, f1_dt, p_soft_dt, r_soft_dt, f1_soft_dt
    

#%%

def run_dt_2_phase(file_path_s,file_path_g,res_output_path):

    y_eval, y_pred_eval, dt_clf = phase_detection_decision_tree(file_path_s,file_path_g)
    
    df_res=result_df(y_eval, y_pred_eval)
    
    p_dt, r_dt, f1_dt, p_soft_dt, r_soft_dt, f1_soft_dt = get_metrics(df_res)
    
    df=pd.DataFrame({})
    
    df["p_dt"], df["r_dt"], df["f1_dt"], df["p_soft_dt"], df["r_soft_dt"], df["f1_soft_dt"], df["clf_size"], df["clf_depth"] = \
        [p_dt], [r_dt], [f1_dt], [p_soft_dt], [r_soft_dt], [f1_soft_dt],  dt_clf.tree_.node_count, dt_clf.get_depth()
    
    pd.DataFrame(df).to_csv(res_output_path,header=1, index=False, sep=" ") 

#%%
if __name__ == '__main__':
    
    file_path_s="/home/pengmiao/Disk/work/data/Graph_dataset/IPDPS/GPOP/LoadTraces/pr.amazon.trace.scatter.txt"
    file_path_g="/home/pengmiao/Disk/work/data/Graph_dataset/IPDPS/GPOP/LoadTraces/pr.amazon.trace.gather.txt"

    res_output_path = "./test_output.csv"

    run_dt_2_phase(file_path_s,file_path_g,res_output_path)


