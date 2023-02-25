from .bert_LMCL_soft import *
from .base_bert import *
from .bert__AAML import *
from .bert_LMCL import *
from .bert_sphere import *
from .bert_AAML_soft import *
from .bert_sphere_soft import *
from .bert_MAAML import *
from .bert_MLMCL import *
from .multilabel_base_bert import *
import torch
from torch import optim

model_classes = {
        'base_bert': base_bert,
        'bert_AAML':bert_AAML,
        'bert_LMCL':bert_LMCL,
        'bert_sphere':bert_sphere,
        "bert_LMCL_soft":bert_LMCL_soft,
        "bert_AAML_soft":bert_AAML_soft,
        "bert_sphere_soft":bert_sphere_soft,
        "bert_MAAML":bert_MAAML,
        "bert_MLMCL":bert_MLMCL,
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