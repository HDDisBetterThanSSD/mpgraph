import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'#'2 in tarim'
import warnings
warnings.filterwarnings('ignore')
import sys
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import pandas as pd
import config as cf
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from model import TMAP
#from preprocessing import read_load_trace_data,preprocessing_patch
from torch.autograd import Variable
from sklearn.metrics import roc_curve,f1_score,recall_score,precision_score,accuracy_score
import lzma
from tqdm import tqdm
from data_loader import data_generator
import os
import pdb
from validation import run_val

if __name__ == "__main__":

    file_path=sys.argv[1]
    NUM= int(sys.argv[2])
    data_generator(file_path,NUM,Read_Pickle=False)