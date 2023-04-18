import config as cf
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device=cf.device

def check_hit(future_list,pred):
    if pred in future_list:
        return 1
    else:
        return 0

def run_validation(model, dataloader, df, vocab,model_save_path, CHECK_FOWARD):
    prediction=[]
    print("model_save_path:",model_save_path)
    val_res_path=model_save_path+".val_res.csv"
    app_name=model_save_path.split("/")[-1].split("-")[0]
    
    model.to(device)
    model.eval()
    for data,ip,lab in tqdm(dataloader):
        output= model(data,ip).cpu()
        prediction.extend([x.topk(1)[1].item() for x in output])
        
    
    df["pred"]=[cf.DELTA_BOUND-i if i>cf.DELTA_BOUND else i for i in prediction][0:len(df)]
    
    df['hit']=df.apply(lambda x: check_hit(x['future_deltas'],x["pred"]),axis=1)
    accuracy=df['hit'].values.sum()/len(df)
    print("accuracy:",accuracy)
    
    df_res={}

    df_res["app"],df_res["acc_10"],df_res["vocab"]=[app_name],[accuracy],[vocab]
    

    pd.DataFrame(df_res).to_csv(val_res_path,header=1, index=False, sep=" ") 
    
    print("Done: results saved at:", val_res_path)
    print(df_res)
    
    return