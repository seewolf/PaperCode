from transformers import BertModel,RobertaModel
from torch import nn
from .multi_label_loss_function import *


class multilabel_base_bert(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.opt=opt
        self.device=opt.device
        if opt.pretrained_bert_name=="bert-base-uncased":
            self.bert= BertModel.from_pretrained(opt.pretrained_bert_name)
        elif opt.pretrained_bert_name in ["roberta-base" ,"roberta-large"]:
            self.bert= RobertaModel.from_pretrained(opt.pretrained_bert_name)
        self.dropout=nn.Dropout(opt.dropout)
        self.linear=nn.Sequential(
            nn.Linear(opt.bert_dim,opt.bert_dim//2),
            nn.Tanh(),
            nn.Linear(opt.bert_dim//2,opt.bert_dim//4),
            nn.Tanh(),
            nn.Linear(opt.bert_dim//4,36)
            )
        self.loss=multi_label_loss_base(self.opt,36,opt.polarities_dim[opt.dataset])
        
    def forward(self,input_ids,token_type_ids,attention_mask,labels=None):
        if self.opt.pretrained_bert_name=="bert-base-uncased":
            outputs=self.bert(input_ids,token_type_ids,attention_mask)
        elif self.opt.pretrained_bert_name in ["roberta-base","roberta-large"]:
             outputs=self.bert(input_ids,attention_mask)
        outputs=self.dropout(outputs.pooler_output)
        outputs=self.linear(outputs)
        outputs,loss=self.loss(outputs,labels)
        
        return outputs,loss