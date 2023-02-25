from trainer import *
from option import *
import numpy as np
import random
import torch


if __name__ == '__main__':
    
    opt=option()
    opt['device']="cuda" if torch.cuda.is_available() else "cpu"
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if opt.dataset=="sst-5":
        t=sst5_trainer(opt)
    elif opt.dataset=="goemotions":
        t=goemotions_trainer(opt)
    elif opt.dataset=="semeval18":
        t=semeval18_trainer(opt)
    t.run()