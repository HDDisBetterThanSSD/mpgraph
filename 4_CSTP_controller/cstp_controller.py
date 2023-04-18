import os
import warnings
warnings.filterwarnings('ignore')
import torch
import numpy as np
import pandas as pd
import sys
from einops import rearrange
from page_base_offset_table import Page_Base_Offset_Table

#ROOT = "/home/pengmiao/Disk/Lab/SC23/Draft/"
ROOT = "../../"
SP_DEGREE=2
TP_DEGREE=2
PBOT_SIZE=128

PBOT=Page_Base_Offset_Table(PBOT_SIZE)

current_dir = os.getcwd()


with open(ROOT+"2_ChampSim_MPG/0_my_scripts/temp.txt", 'r') as file:
    file_contents = file.read()

APP_NAME=file_contents.split(" ")[-1][:-6]
PHASE=file_contents.split(" ")[-2]

#%% Spatial predictor loading

SP_MODEL_LOAD_PATH=ROOT+'res/2_Spatial_delta_predictor/'+ APP_NAME +PHASE +'.pth'
SP_ROOT=ROOT+'3_DS_Models/2_Spatial_delta_predictor/'
sys.path.append(SP_ROOT)
sys.path.append(SP_ROOT+'/SP_AMMA/')

import SP_AMMA.config as sp_cf
import SP_AMMA.model as sp_model
from SP_AMMA.preprocessing import preprocessing_gen as sp_preprocessing_gen

device=sp_cf.device
batch_size=sp_cf.batch_size
epochs = sp_cf.epochs
lr = sp_cf.lr
gamma = sp_cf.gamma
step_size=sp_cf.step_size
pred_num=sp_cf.PRED_FORWARD
early_stop = sp_cf.early_stop
DELTA_BOUND=sp_cf.DELTA_BOUND
BITMAP_SIZE=sp_cf.BITMAP_SIZE

sp_delta_model = sp_model.TMAP_ATM(
    image_size=sp_cf.image_size,
    patch_size=sp_cf.patch_size,
    num_classes=sp_cf.num_classes,
    dim=sp_cf.dim,
    depth=sp_cf.depth,
    heads=sp_cf.heads,
    mlp_dim=sp_cf.mlp_dim,
    channels=sp_cf.channels,
    context_gamma=sp_cf.context_gamma
).to(device)


sp_delta_model.load_state_dict(torch.load(SP_MODEL_LOAD_PATH))



print("Load model:", SP_MODEL_LOAD_PATH)


def sp_outut_to_pref(block_address,pred_index):
    if pred_index<DELTA_BOUND:
        pred_delta=pred_index+1
    else:
        pred_delta=pred_index-BITMAP_SIZE
#    print("spatial index:",pred_index)
#    print("spatial delta:",pred_delta)
    return (block_address+pred_delta)<<6

def sp_model_predict(input_Addr,input_PC):
    df=pd.DataFrame({})    
    df["addr"],df["ip"],df["id"],df["cycle"],df["hit"]=input_Addr,input_PC,0,0,0
    df_res=sp_preprocessing_gen(df)
    past_b=df_res.past.values.tolist()
    past_ip_b = df_res.past_ip.values.tolist()
    past_page_b =df_res.past_page.values.tolist()

    data=rearrange(np.array(past_b), '(b c) h w-> b c h w',c=sp_cf.channels,h=sp_cf.image_size[0],w=sp_cf.image_size[1])
    past_tensor=torch.Tensor(data).to(device)
    past_ip_tensor=torch.Tensor(past_ip_b).to(device)
    past_page_tensor=torch.Tensor(past_page_b).to(device)

    output= sp_delta_model(past_tensor,past_ip_tensor,past_page_tensor).cpu()[0]

    pred_index_list=output.topk(SP_DEGREE)[1].numpy()
    block_address=(input_Addr[-1]>>6)
    if len(pred_index_list)>0:
        pref_list=[sp_outut_to_pref(block_address,x) for x in pred_index_list]
    else:
        pref_list=(block_address+1)<<6
    return pref_list


#%% Temporal predictor loading

TP_MODEL_LOAD_PATH=ROOT+'res/3_Temporal_page_predictor/'+ APP_NAME +PHASE +'.pth'
TP_ROOT=ROOT+'3_DS_Models/3_Temporal_page_predictor/'
sys.path.append(TP_ROOT)
sys.path.append(TP_ROOT+'/TP_AMMA/')

import TP_AMMA.config as tp_cf
import TP_AMMA.model as tp_model
import json

with open(TP_MODEL_LOAD_PATH[:-3]+'token_dict_p2i.json', 'r') as f:
    token_dict_p2i_c= json.load(f)
    token_dict_p2i={int(k): int(v) for k, v in token_dict_p2i_c.items()}

with open(TP_MODEL_LOAD_PATH[:-3]+'token_dict_i2p.json', 'r') as f:
    token_dict_i2p_c= json.load(f)
    token_dict_i2p={int(k): int(v) for k, v in token_dict_i2p_c.items()}

vocab = len(token_dict_p2i)
LOOK_BACK=tp_cf.LOOK_BACK
device=tp_cf.device
batch_size=tp_cf.batch_size
epochs = tp_cf.epochs
lr = tp_cf.lr
gamma = tp_cf.gamma
step_size=tp_cf.step_size
pred_num=tp_cf.PRED_FORWARD
early_stop = tp_cf.early_stop


tp_page_model = tp_model.TMAP_Page(
    input_length=tp_cf.HISTORY,
    embedding_dim=tp_cf.dim,
    dim=tp_cf.dim,
    dim_head=tp_cf.dim,
    depth=tp_cf.depth,
    heads=tp_cf.heads,
    vocab=vocab,
    out_dim=tp_cf.BITMAP_SIZE
).to(device)


tp_page_model.load_state_dict(torch.load(TP_MODEL_LOAD_PATH))

print("Load model:", TP_MODEL_LOAD_PATH)


def page_to_token(x, token_dict_p2i):
    if x in token_dict_p2i.keys():
        return token_dict_p2i[x]
    else:
        return 0

def tp_outut_to_pref(page_address,pred_index):
    if pred_index<DELTA_BOUND:
        pred_delta=pred_index
    else:
        pred_delta=tp_cf.DELTA_BOUND-pred_index
    return page_address+pred_delta

def tp_model_predict(input_Addr,input_PC):
    df=pd.DataFrame({})    
    df["addr"],df["ip"],df["id"],df["cycle"],df["hit"]=input_Addr,input_PC,0,0,0
    #df_res=tp_process_token_vocab_gen(df,token_dict_p2i,token_dict_i2p)
    df_res=sp_preprocessing_gen(df)
    past_ip_b = df_res.past_ip.values.tolist()
    past_page_b = [int(x) for x in df_res.past_page_abs.values.tolist()[0]]
    past_page_token=[[page_to_token(x,token_dict_p2i) for x in past_page_b]]
    
    past_ip_tensor=torch.Tensor(past_ip_b).to(device) 
    past_page_tensor=torch.IntTensor(past_page_token).to(device)

    output= tp_page_model(past_page_tensor,past_ip_tensor).cpu()[0]
    
    pred_page_delta=output.topk(1)[1].numpy()#TP_DEGREE
    
    page_address=past_page_b[-1]
    if len(pred_page_delta)>0:
        pref_page=[tp_outut_to_pref(page_address,x) for x in pred_page_delta]
    else:
        pref_page=(page_address+1)
    return pref_page
    

#%%

def update_pbot(input_Addr,input_PC,PBOT):
    page_addr= [ x >> sp_cf.PAGE_BITS for x in input_Addr]
    block_offset= [ (x >> sp_cf.BLOCK_BITS) - ((x >> sp_cf.PAGE_BITS) << (sp_cf.PAGE_BITS-sp_cf.BLOCK_BITS )) for x in input_Addr]
    
    for i,page in enumerate(page_addr):
        PBOT[page]=(block_offset[i],input_PC[i])

    return PBOT

def cstp_controller(input_Addr,input_PC):
    update_pbot(input_Addr,input_PC,PBOT)
    pref_list=sp_model_predict(input_Addr,input_PC)
    pred_page=tp_model_predict(input_Addr,input_PC)[0]
    
    input_new=input_Addr
    pc_new = input_PC
    for i in range(TP_DEGREE):
        if pred_page in PBOT.dict:
            offset=PBOT[pred_page][0]
            new_addr=((pred_page << (sp_cf.PAGE_BITS-sp_cf.BLOCK_BITS))+offset)<<sp_cf.BLOCK_BITS
            new_pc =PBOT[pred_page][1]
            input_new=input_new[1:]+[new_addr]
            pc_new=pc_new[1:]+[new_pc]
            pref_list.extend(sp_model_predict(input_new,pc_new))
            pred_page=tp_model_predict(input_new,pc_new)[0]
        else:
            pass
    #print(pref_list)
    return pref_list


