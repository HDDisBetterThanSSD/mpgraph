
import config as cf
import logging
import numpy as np
import pandas as pd
import torch
import pdb
import lzma
import itertools
from tqdm import tqdm
import os


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
device=cf.device


def read_load_trace_data(load_trace, num_prefetch_warmup_instructions,num_total_instructions,skipping=0):
    
    def process_line(line):
        split = line.strip().split(', ')
        return int(split[0]), int(split[1]), int(split[2], 16), int(split[3], 16), split[4] == '1'

    train_data = []
    eval_data = []
    if load_trace[-2:] == 'xz':
        with lzma.open(load_trace, 'rt') as f:
            for line in f:
                pline = process_line(line)
                if pline[0]>skipping*1000000:
                    if pline[0] < num_prefetch_warmup_instructions * 1000000:
                        train_data.append(pline)
                    else:
                        if pline[0] < num_total_instructions * 1000000:
                            eval_data.append(pline)
                        else:
                            break
    else:
        with open(load_trace, 'r') as f:
            for line in f:
                pline = process_line(line)
                if pline[0]>skipping*1000000:
                    if pline[0] < num_prefetch_warmup_instructions * 1000000:
                        train_data.append(pline)
                    else:
                        if pline[0] < num_total_instructions * 1000000:
                            eval_data.append(pline)
                        else:
                            break

    return train_data, eval_data

def page_to_token(x, token_dict_p2i):
    if x in token_dict_p2i.keys():
        return token_dict_p2i[x]
    else:
        return -1


def addr_hash(x,HASH_BITS):
    t = int(x)^(int(x)>>32); 
    result = (t^(t>>HASH_BITS)) & (2**HASH_BITS-1); 
    return result/(2**HASH_BITS)

def ip_list_norm(ip_list,HASH_BITS):
    return [addr_hash(ip,HASH_BITS) for ip in ip_list]

def page_list_norm(page_list,current_page):
    return list(1/(abs(np.array(page_list)-current_page)+1))
     
def delta_acc_list(delta,DELTA_BOUND=128):#delta accumulative list
    res=list(itertools.accumulate(delta))
    res=[i for i in res if abs(i)<=DELTA_BOUND]
    if len(res)==0:
        res="nan"
    return res

def delta_min_abs(delta,DELTA_BOUND=128):#delta accumulative list
    if delta=="nan":
        return "nan"
    delta2=list(filter(lambda x: x != 0, delta))
    if len(delta2)==0:
        return "nan"
    res=min(delta2, key=abs)
    if abs(res)<DELTA_BOUND:
        if res<0:
            res=DELTA_BOUND-res #turn -1 to 129
            # turn back 128-pred_delta: 128-129=-1
        return res
    else:
        return "nan"

def process_token_vocab(data):
    print("preprocessing with context")
    df=pd.DataFrame(data)
    df.columns=["id", "cycle", "addr", "ip", "hit"]
    df['raw']=df['addr']
    df['block_address'] = [x>>BLOCK_BITS for x in df['raw']]
    df['page_address'] = [ x >> PAGE_BITS for x in df['raw']]
    #df['page_address_str'] = [ "%d" % x for x in df['page_address']]
    df['page_offset'] = [x- (x >> PAGE_BITS<<PAGE_BITS) for x in df['raw']]
    df['block_index'] = [int(x>> BLOCK_BITS) for x in df['page_offset']]  
    #df["block_address_bin"]=df.apply(lambda x: convert_to_binary(x['block_address'],BLOCK_NUM_BITS),axis=1)
    df['page_addr_delta']=df['page_address'].diff()
    
    # past
    for i in range(LOOK_BACK):
        df['ip_past_%d'%(i+1)]=df['ip'].shift(periods=(i+1))
        #df['page_past_%d'%(i+1)]=df['page_address'].shift(periods=(i+1))
    

    past_ip_name=['ip_past_%d'%(i) for i in range(LOOK_BACK,0,-1)]
    #past_page_name=['page_past_%d'%(i) for i in range(LOOK_BACK,0,-1)]
    past_ip_name.append('ip')
    #past_page_name.append('page_address')
    #Pem, update done
    
    df["past_ip_abs"]=df[past_ip_name].values.tolist()
    #df["past_page_abs"]=df[past_page_name].values.tolist()
    
    df=df.dropna()
    
    
    df['past_ip']=df.apply(lambda x: ip_list_norm(x['past_ip_abs'],16),axis=1)
    #df['past_page']=df.apply(lambda x: page_list_norm(x['past_page_abs'],x['page_address']),axis=1)
   
    
    #labels
    '''
    future_idx: delta to the prior addr
    future_delta: accumulative delta to current addr
    '''
    for i in range(PRED_FORWARD):
        df['page_future_%d'%(i+1)]=df['page_addr_delta'].shift(periods=-(i+1))
    
    for i in range(PRED_FORWARD):
            if i==0:
                df["future_idx"]=df[['page_future_%d'%(i+1)]].values.astype(int).tolist()
            else:   
                #df["future_idx"] = df[['future_idx','delta_future_%d'%(i+1)]].values.astype(int).tolist()
                df["future_idx"]=np.hstack((df["future_idx"].values.tolist(), df[['page_future_%d'%(i+1)]].values.astype(int))).tolist()
    
    
    page_list=df.page_address.unique()
    page_list.sort()
    
    print("Tokenizing")

    page_token_list=page_list.tolist()
    vocab=len(page_token_list)
    token_dict_p2i={k: v for v, k in enumerate(page_token_list)}
    token_dict_i2p={v: k for v, k in enumerate(page_token_list)}
    
    vocab=len(page_token_list)
    
    df["future_deltas"]=df.apply(lambda x: delta_acc_list(x['future_idx'],DELTA_BOUND),axis=1)
    df=df.dropna()
    
    df["future"]=df.apply(lambda x: delta_min_abs(x['future_deltas'],DELTA_BOUND),axis=1)
    
    df=df[df["future"]!="nan"]
    
    
    df['page_token']=df.apply(lambda x: page_to_token(x['page_address'],token_dict_p2i),axis=1)
    
    df = df[df['page_token']>-1]
    
    
    df=df.dropna()
    df=df[['id', 'cycle', 'addr', 'ip', 'hit', 'raw', 'block_address', 'page_addr_delta',
       'page_address', 'page_token','past_ip','future_deltas','future']]
    
    return df, vocab, token_dict_p2i,token_dict_i2p


def preprocessing(df, vocab):
    for i in range(LOOK_BACK):
        #df['ip_past_%d'%(i+1)]=df['ip'].shift(periods=(i+1))
        df['page_past_%d'%(i+1)]=df['page_token'].shift(periods=(i+1))
    
    past_page_name=['page_past_%d'%(i) for i in range(LOOK_BACK,0,-1)]   
    past_page_name.append('page_token')
    df["past"]=df[past_page_name].values.astype(int).tolist()
    
    df =df.dropna()
    
    return df[['id', 'cycle', 'ip', 'past_ip','page_token','past','future_deltas','future','page_addr_delta']]


    
    
    
#%% gen

def process_token_vocab_gen(data,token_dict_p2i,token_dict_i2p):
    print("preprocessing with context")
    df=pd.DataFrame(data)
    df.columns=["id", "cycle", "addr", "ip", "hit"]
    df['raw']=df['addr']
    df['block_address'] = [x>>BLOCK_BITS for x in df['raw']]
    df['page_address'] = [ x >> PAGE_BITS for x in df['raw']]
    #df['page_address_str'] = [ "%d" % x for x in df['page_address']]
    df['page_offset'] = [x- (x >> PAGE_BITS<<PAGE_BITS) for x in df['raw']]
    df['block_index'] = [int(x>> BLOCK_BITS) for x in df['page_offset']]  
    #df["block_address_bin"]=df.apply(lambda x: convert_to_binary(x['block_address'],BLOCK_NUM_BITS),axis=1)
    #df['page_addr_delta']=df['page_address'].diff()
    
    # past
    for i in range(LOOK_BACK):
        df['ip_past_%d'%(i+1)]=df['ip'].shift(periods=(i+1))

    past_ip_name=['ip_past_%d'%(i) for i in range(LOOK_BACK,0,-1)]

    past_ip_name.append('ip')

    df["past_ip_abs"]=df[past_ip_name].values.tolist()
    #df["past_page_abs"]=df[past_page_name].values.tolist()
    
    #df=df.dropna()
    
    #df['past_ip']=df.apply(lambda x: ip_list_norm(x['past_ip_abs'],16),axis=1)

    
    #df['page_token']=df.apply(lambda x: page_to_token(x['page_address'],token_dict_p2i),axis=1)
    
    #df = df[df['page_token']>-1]
    
    
    #df=df.dropna()
    #df=df[['id', 'cycle', 'addr', 'ip', 'hit', 'raw', 'block_address',
    #   'page_address', 'page_token','past_ip']]
    
    return df