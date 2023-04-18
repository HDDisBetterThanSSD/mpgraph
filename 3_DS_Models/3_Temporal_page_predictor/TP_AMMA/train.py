import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import config as cf
import warnings
import os
from tqdm import tqdm
from data_loader import data_generator
from torch.optim.lr_scheduler import ExponentialLR
import torch.optim as optim
from torch.autograd import Variable
from validation import run_validation
from torchinfo import summary
from model import TMAP_Page,TMAP_Page_2

import json

warnings.filterwarnings('ignore')
LOOK_BACK=cf.LOOK_BACK
device=cf.device
batch_size=cf.batch_size
epochs = cf.epochs
lr = cf.lr
gamma = cf.gamma
step_size=cf.step_size
pred_num=cf.PRED_FORWARD
early_stop = cf.early_stop

log=cf.Logger()
    
#TRAIN_NUM=2
#TOTAL_NUM = 4
#SKIP_NUM = 0

#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

#%%

#pdb.set_trace()
def train(model,optimizer,ep,train_loader):
    epoch_loss = 0
    model.train()
    for batch_idx, (data, ip, target) in enumerate(train_loader):#d,t: (torch.Size([64, 1, 784]),64)        
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data,ip)
        loss =F.cross_entropy(output,target)
        #loss = F.binary_cross_entropy(output, target,reduction='mean')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss/=len(train_loader)
    return epoch_loss


def test(model,test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, ip, target in test_loader:
            data, target = Variable(data), Variable(target)
            output = model(data,ip)
            loss =F.cross_entropy(output,target)
            test_loss += loss.item()
        test_loss/=len(test_loader)
        return test_loss

def run_epoch(model,optimizer,epochs, loading, model_save_path,train_loader,test_loader,lr):
    best_loss=0
    early_stop=cf.early_stop
    model.to(device)
    for epoch in range(epochs):
        train_loss=train(model,optimizer,epoch,train_loader)
        test_loss=test(model,test_loader)
        log.logger.info((f"Epoch: {epoch+1} - loss: {train_loss:.10f} - test_loss: {test_loss:.10f}"))
        if epoch == 0:
            best_loss=test_loss
        if test_loss<=best_loss:
            torch.save(model.state_dict(), model_save_path)    
            best_loss=test_loss
            log.logger.info("-------- Save Best Model! --------")
            early_stop=cf.early_stop
        else:
            early_stop-=1
            log.logger.info("Early Stop Left: {}".format(early_stop))
        if early_stop == 0:
            log.logger.info("-------- Early Stop! --------")
            break


#%%

def train_main(file_path_s,file_path_g,output_path,TOTAL_NUM=20,TRAIN_NUM=0,SKIP_NUM=0,PHASE="mix"):
    
    model_save_path=output_path+"."+PHASE+".pth"
    log_path=model_save_path+".log"
    log.set_logger(log_path)
    log.logger.info("%s"%output_path)
    
    train_dataloader, df_train, dev_dataloader, df_test, vocab, token_dict_p2i,token_dict_i2p= \
    data_generator(file_path_s,file_path_g,TOTAL_NUM,0,0)

    #store token
    with open(model_save_path[:-3]+'token_dict_p2i.json', 'w') as f:
        json.dump(token_dict_p2i, f)
    with open(model_save_path[:-3]+'token_dict_i2p.json', 'w') as f:
        json.dump(token_dict_i2p, f)


    model = TMAP_Page(
        input_length=cf.HISTORY,
        embedding_dim=cf.dim,
        dim=cf.dim,
        dim_head=cf.dim,
        depth=cf.depth,
        heads=cf.heads,
        vocab=vocab,
        out_dim=cf.BITMAP_SIZE
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=cf.lr)
    
    log.logger.info(summary(model))
    
    run_epoch(model,optimizer,epochs, False,model_save_path,train_dataloader,dev_dataloader,lr=cf.lr)
    
    run_validation(model,dev_dataloader,df_test,vocab,model_save_path, CHECK_FOWARD=10)
    log.shutdown()


if __name__ == "__main__":
    file_path_s="/home/pengmiao/Disk/work/data/Graph_dataset/IPDPS/GPOP/LoadTraces/cc.amazon.trace.scatter.txt"
    file_path_g="/home/pengmiao/Disk/work/data/Graph_dataset/IPDPS/GPOP/LoadTraces/cc.amazon.trace.gather.txt"
    output_path="./res.csv"
    train_main(file_path_s,file_path_g,output_path)

#%%
'''
#%% Model Size
from model import TMAP_Page,TMAP_Page_2
DIMENSION=64
HEADS=4

HISTORY=9
LENGTH=9
VOCAB=2

#original size
model = TMAP_Page(
    input_length=LENGTH,
    embedding_dim=DIMENSION,
    dim_head=DIMENSION,
    dim=DIMENSION,
    depth=1,
    heads=HEADS,
    vocab=65536,
    out_dim=65536
).to(device)
print(summary(model))

# binary-encoding size
model = TMAP_Page_2(
    input_length=LENGTH,
    embedding_dim=DIMENSION,
    dim_head=DIMENSION,
    dim=DIMENSION,
    depth=1,
    heads=HEADS,
    vocab=VOCAB,
    out_dim=16
).to(device)
print(summary(model))
'''
