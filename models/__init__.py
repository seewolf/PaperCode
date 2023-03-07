
from .multilabel_base_bert import *
import torch
from torch import optim

model_classes = {
        "multilabel_base_bert":multilabel_base_bert
        }

optim_classes={
        'adam':optim.Adam,
        'adamw':optim.AdamW
        }

initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }