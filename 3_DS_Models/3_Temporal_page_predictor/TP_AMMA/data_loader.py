from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset,DataLoader
from preprocessing import read_load_trace_data,preprocessing
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import config as cf
import numpy as np
import torch
from preprocessing import process_token_vocab, read_load_trace_data
from sklearn.model_selection import train_test_split

device = cf.device
batch_size=cf.batch_size

class MAPDataset(Dataset):
    def __init__(self, df):
        self.past=list(df["past"].values)
        self.future=list(df["future"].values)
        self.past_ip=list(df["past_ip"].values)

    def __getitem__(self, idx):
        
        past = self.past[idx]
        future = self.future[idx]
        past_ip = self.past_ip[idx]

        return [past,past_ip,future]

    def __len__(self):
        return len(self.past)
    
    
    def collate_fn(self, batch):
        
        past_b = [x[0] for x in batch]
        past_ip_b = [x[1] for x in batch]
        future_b = [x[2] for x in batch]
#        data=rearrange(past_b, '(b h c) w-> b c h w',c=cf.channels,b=batch,h=cf.image_size[0],w=cf.image_size[1])
#        print(np.array(past_b).shape)
        data=np.array(past_b)
        #data=rearrange(np.array(past_b), '(b c) h w-> b c (h w)',c=cf.channels,h=LOOK_BACK+1,w=cf.image_size[1])

        past_tensor=torch.IntTensor(data).to(device)
        
        future_tensor=torch.tensor(future_b,dtype=torch.int64).to(device)
        
        past_ip_tensor=torch.Tensor(past_ip_b).to(device)
        
        return past_tensor, past_ip_tensor, future_tensor

def data_generator(file_path_s,file_path_g,TRAIN_NUM, TOTAL_NUM,SKIP_NUM=0):
    train_data_s, eval_data_s = read_load_trace_data(file_path_s, TRAIN_NUM,TOTAL_NUM,SKIP_NUM)
    train_data_g, eval_data_g = read_load_trace_data(file_path_g, TRAIN_NUM,TOTAL_NUM,SKIP_NUM)
    
    all_data=train_data_s + eval_data_s + train_data_g + eval_data_g
    
    df, vocab, token_dict_p2i,token_dict_i2p =process_token_vocab(all_data)
    print("vocab:",vocab)
    df=preprocessing(df,vocab)
        
    df_train, df_test = train_test_split(df, test_size=0.5)
    train_dataset = MAPDataset(df_train)
    test_dataset = MAPDataset(df_test)
    
    train_dataloader=DataLoader(train_dataset,batch_size=cf.batch_size,shuffle=True,collate_fn=train_dataset.collate_fn)
    dev_dataloader=DataLoader(test_dataset,batch_size=cf.batch_size,shuffle=False,collate_fn=test_dataset.collate_fn)
    
    return train_dataloader, df_train, dev_dataloader, df_test, vocab, token_dict_p2i,token_dict_i2p