import torch

NUM_THREAD=4
#%%HW Configuration; fixed
BLOCK_BITS=6
PAGE_BITS=12#12
#SUPER_PAGE_BITS=13
TOTAL_BITS=64
BLOCK_NUM_BITS=TOTAL_BITS-BLOCK_BITS

#%% Input and labeling; 
#%%%tunable
SPLIT_BITS=6
HISTORY=9
LOOK_BACK=HISTORY-1 
#LOOK_FORWARD=200#look forward for collect training labels
#PRED_FORWARD=128#pred forward
PRED_FORWARD=10#pred forward
DELTA_BOUND=128


'''
bitmap:e.g. DELTA_BOUND=4; bitmap length = 2*DELTA_BOUND=8
    index: [0,1,2,3, 4, 5, 6, 7]
    value: [1,2,3,4,-4,-3,-2,-1]
value = index+1  ; <DELTA_BOUND
      = index - DELTA_BOUND ; >DELTA_BOUND
'''


#%%% fixed
BITMAP_SIZE=2*DELTA_BOUND
#BITMAP_SIZE=2*DELTA_BOUND+1
image_size=(LOOK_BACK+1,BLOCK_NUM_BITS//SPLIT_BITS+1)#h,w
patch_size=(1,image_size[1])
num_classes=BITMAP_SIZE

#%% filter
#Degree=2
#FILTER_SIZE=10
Degree=16
FILTER_SIZE=16

#%%
#Model; tunable
#%%%shape


dim=16
heads=4

depth=1
mlp_dim=dim
channels=1
context_gamma=0.2
'''

dim=16
depth=2
heads=4
mlp_dim=16
channels=1
context_gamma=0.2
'''


#%%% LSTM
lstm_input_length=image_size[0]*image_size[1]
lstm_hid_dim=128
lstm_depth=1
lstm_out_dim=num_classes

vocab_size=2**SPLIT_BITS

#%%% training
batch_size=16
lr = 1e-2

epochs=20
early_stop=20

#%%% scheduler
gamma = 0.1
step_size=20

gpu_id = '0'
#device_id = [0, 1]
device_id = [0]

# set device
if gpu_id != '':
    device = torch.device(f"cuda:{gpu_id}")
else:
    device = torch.device('cpu')

device = torch.device('cpu')
    
#%% logging
#log_interval=5

import os
import logging
from logging import handlers
class Logger(object):
    
    def __init__(self):
        pass
    
    def set_logger(self, log_path):
        #if os.path.exists(log_path) is True:
        #    os.remove(log_path)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
    
        if not self.logger.handlers:
            # Logging to a file
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
            self.logger.addHandler(file_handler)
    
            # Logging to console
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(stream_handler)
            
    def shutdown(self):
        logging.shutdown()
    
