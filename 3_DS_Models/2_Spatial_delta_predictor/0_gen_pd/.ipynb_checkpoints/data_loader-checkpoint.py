import torch
import json
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset,DataLoader
from preprocessing import read_load_trace_data,preprocessing
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import config as cf
from sklearn.model_selection import train_test_split
import pandas as pd
device = cf.device
batch_size=cf.batch_size
#%%
class MAPDataset(Dataset):
    def __init__(self, df):
        self.past=list(df["past"].values)
        self.future=list(df["future"].values)
        self.past_ip=list(df["past_ip"].values)
        self.past_page=list(df["past_page"].values)

    def __getitem__(self, idx):
        
        past = self.past[idx]
        future = self.future[idx]
        past_ip = self.past_ip[idx]
        past_page = self.past_page[idx]
        return [past,past_ip,past_page,future]

    def __len__(self):
        return len(self.past)
    
    
    def collate_fn(self, batch):
        
        past_b = [x[0] for x in batch]
        past_ip_b = [x[1] for x in batch]
        past_page_b = [x[2] for x in batch]
        future_b = [x[3] for x in batch]
#        data=rearrange(past_b, '(b h c) w-> b c h w',c=cf.channels,b=batch,h=cf.image_size[0],w=cf.image_size[1])
#        print(np.array(past_b).shape)
        data=rearrange(np.array(past_b), '(b c) h w-> b c h w',c=cf.channels,h=cf.image_size[0],w=cf.image_size[1])

        past_tensor=torch.Tensor(data).to(device)
        
        future_tensor=torch.Tensor(future_b).to(device)
        
        past_ip_tensor=torch.Tensor(past_ip_b).to(device)
        past_page_tensor=torch.Tensor(past_page_b).to(device)
        
        return past_tensor, past_ip_tensor,past_page_tensor,future_tensor



def data_generator2(file_path,NUM,Read_Pickle=False,only_val=False):
    data = read_load_trace_data(file_path,NUM)
    df_all = preprocessing(data)
    df_train, df_test = train_test_split(df_all, test_size=0.5)

    df_train_s=df_train[df_train['phase']==1]
    df_test_s=df_test[df_test['phase']==1]

    df_train_g=df_train[df_train['phase']==2]
    df_test_g=df_test[df_test['phase']==2]

    df_train_s.to_pickle(file_path+'.scatter'+".train.pkl")
    print ("output to pickle: ", file_path+'.scatter'+".train.pkl")
    df_test_s.to_pickle(file_path+'.scatter'+".test.pkl")
    print ("output to pickle: ", file_path+'.scatter'+".test.pkl")
    
    df_train_g.to_pickle(file_path+'.gather'+".train.pkl")
    print ("output to pickle: ", file_path+'.gather'+".train.pkl")
    df_test_g.to_pickle(file_path+'.gather'+".test.pkl")
    print ("output to pickle: ", file_path+'.gather'+".test.pkl")

    return


def data_generator(file_path,NUM,Read_Pickle=False,only_val=False):
    data = read_load_trace_data(file_path,NUM)
    df_all = preprocessing(data)
    
    df_s=df_all[df_all['phase']==1]
    df_g=df_all[df_all['phase']==2]
    
    
    df_train_s, df_test_s = train_test_split(df_s, test_size=0.5)
    df_train_g, df_test_g = train_test_split(df_g, test_size=0.5)


    df_train_s.to_pickle(file_path+'.scatter'+".train.pkl")
    print ("output to pickle: ", file_path+'.scatter'+".train.pkl")
    df_test_s.to_pickle(file_path+'.scatter'+".test.pkl")
    print ("output to pickle: ", file_path+'.scatter'+".test.pkl")
    
    df_train_g.to_pickle(file_path+'.gather'+".train.pkl")
    print ("output to pickle: ", file_path+'.gather'+".train.pkl")
    df_test_g.to_pickle(file_path+'.gather'+".test.pkl")
    print ("output to pickle: ", file_path+'.gather'+".test.pkl")

    return
