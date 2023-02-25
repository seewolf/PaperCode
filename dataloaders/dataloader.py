from concurrent.futures import process
import datasets
from .utils import *
from option import option
from torch.utils.data import DataLoader,Dataset
import torch
import torch.nn.functional as F


class dataLoader:
    def __init__(self,opt:option):
        self.opt=opt
        self.vaild_datasets={"sst-5":"SetFit/sst5","goemotions":"go_emotions","semeval18":"sem_eval_2018_task_1"}
        if self.opt.dataset not in self.vaild_datasets.keys():
            raise Exception('unknown dataset!')
        self.tokenizer=Tokenizer4Bert(opt.max_seq_len,opt.pretrained_bert_name)
        self.train_data,self.valid_data,self.test_data=self.build()
        self.train_loader=DataLoader(self.train_data,batch_size=self.opt.batch_size,shuffle=True,collate_fn=self.collate_fn)
        self.valid_loader=DataLoader(self.valid_data,batch_size=self.opt.batch_size,shuffle=True,collate_fn=self.collate_fn)
        self.test_loader=DataLoader(self.test_data,batch_size=self.opt.batch_size,shuffle=True,collate_fn=self.collate_fn)
        



    def build(self):
        dataset=self.opt['dataset']
        train_data,valid_data,test_data=None,None,None
        p=None
        if dataset=='sst-5':
            datafiles={'train':"./data/sst_data//train.csv","dev":"./data/sst_data/dev.csv","test":"./data/sst_data/test.csv"}
            train_data,valid_data,test_data=datasets.load_dataset("csv",data_files=datafiles,split=['train','dev','test'])
            p=preprocess
            
        if dataset=='goemotions':
            train_data,valid_data,test_data=datasets.load_dataset("go_emotions",split=["train","validation",'test'])
            p=go_emotion_preprocess
            
        if dataset=="semeval18":
            train_data,valid_data,test_data=datasets.load_dataset("sem_eval_2018_task_1",'subtask5.english',split=["train","validation",'test'])
            p=semeval18_preprocess
            
            
        train_data=train_data.map(p,fn_kwargs={"tokenizer":self.tokenizer},load_from_cache_file=True)
        valid_data=valid_data.map(p,fn_kwargs={"tokenizer":self.tokenizer},load_from_cache_file=True)
        test_data=test_data.map(p,fn_kwargs={"tokenizer":self.tokenizer},load_from_cache_file=True)
        return train_data,valid_data,test_data
    
    
    def collate_fn(self,batch):
        dataset=self.opt['dataset']
        if dataset=='sst-5':
            return self._sst5_collate_fn(batch)
        elif dataset=="goemotions":
            return  self._goemotions_collate_fn(batch)
        elif dataset=="semeval18":
            return  self._semeval18_collate_fn(batch)
    
    def _sst5_collate_fn(self,batch):
        input_ids   = torch.tensor([i['input_ids'] for i in batch])
        token_type_ids   = torch.tensor([i['token_type_ids'] for i in batch])
        attention_mask   = torch.tensor([i['attention_mask'] for i in batch])
        labels     = torch.tensor([i['label'] for i in batch])
        onehot_labels=F.one_hot(labels,self.opt.polarities_dim[self.opt.dataset])
        batch={
                'input_ids': input_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
                'labels':labels
        }
        return batch
    
    def _goemotions_collate_fn(self,batch):
        input_ids   = torch.tensor([i['input_ids'] for i in batch])
        token_type_ids   = torch.tensor([i['token_type_ids'] for i in batch])
        attention_mask   = torch.tensor([i['attention_mask'] for i in batch])
        labels=torch.zeros((input_ids.shape[0],self.opt.polarities_dim[self.opt.dataset]))
        for i in range(labels.shape[0]):
            for j in batch[i]['labels']:
                labels[i,j]=1
        return {
                'input_ids': input_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
                'labels':labels
        }
        
    def _semeval18_collate_fn(self,batch):
        input_ids   = torch.tensor([i['input_ids'] for i in batch])
        token_type_ids   = torch.tensor([i['token_type_ids'] for i in batch])
        attention_mask   = torch.tensor([i['attention_mask'] for i in batch])
        labels     = torch.tensor([i['labels'] for i in batch])
        return {
                'input_ids': input_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
                'labels':labels
        }
    
    






        
    

    
    