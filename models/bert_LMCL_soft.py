from .base_bert import base_bert
from .loss_functions import  AngularPenaltySMLossWithSoftLabel
from torch import nn as nn



class bert_LMCL_soft(base_bert):
    def __init__(self,opt):
        super().__init__(opt)
        self.adms_loss = AngularPenaltySMLossWithSoftLabel(36,self.opt.polarities_dim[opt.dataset], loss_type='cosface',s=opt.s,m=opt.m,k=opt.k).to(self.device)
        self.linear=nn.Sequential(
            nn.Linear(opt.bert_dim,opt.bert_dim//2),
            nn.GELU(),
            nn.Linear(opt.bert_dim//2,opt.bert_dim//4),
            nn.GELU(),
            nn.Linear(opt.bert_dim//4,36)
            )
        
    def forward(self,input_ids,token_type_ids,attention_mask,labels):
        if self.opt.pretrained_bert_name=="bert-base-uncased":
            outputs=self.bert(input_ids,token_type_ids,attention_mask)
        elif self.opt.pretrained_bert_name in["roberta-base","roberta-large"]:
             outputs=self.bert(input_ids,attention_mask)
        outputs=self.dropout(outputs.last_hidden_state[:,0])
        outputs=self.linear(outputs)
        return self.adms_loss(outputs,labels)
        
        