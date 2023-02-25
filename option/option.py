import argparse
import torch


class option(dict):
    def __init__(self):
        super().__init__()
        self.optional={
                "model_name": ["base_bert","bert_AAML","bert_LMCL","bert_sphere",
                               "bert_LMCL_soft","bert_AAML_soft","bert_sphere_soft",
                               "bert_MAAML","bert_MLMCL","bert_Msphere","multilabel_base_bert"],
                "dataset": ["sst-5","goemotions","semeval18"],
                "optimizer": ["adam","adamw"],
                "initializer":["xavier_uniform_","xavier_normal_","orthogonal_"],
                "learning_rate": 1e-5,
                "dropout": 0.1,
                "l2reg": 1e-5,
                "num_epoch": 8,
                "valid_size":0.15,
                "batch_size": 32,
                "bert_dim": 768,
                "pretrained_bert_name": ["bert-base-uncased","roberta-base","microsoft/deberta-v3-base","microsoft/deberta-base","roberta-large"],
                "max_seq_len": 256,
                "polarities_dim": {"sst-5":5,"goemotions":28,"semeval18":11},
                "seed":2022,
                "s":32,
                "m":0.2,
                "k":2.0,
                "em_num_of_perclass":3,
                "inner_margin":0.5,
                "threshold":0.3
            }

        self.parser = argparse.ArgumentParser()

        for key,value in self.optional.items():
            temp=value[0] if isinstance(value,list) else value
            if isinstance(value,list):
                self.parser.add_argument("--"+key,default=temp,type=type(temp),choices=value)
            else:
                self.parser.add_argument("--"+key,default=temp,type=type(temp))
        
        self.dic=vars(self.parser.parse_args())

    def __getattr__(self,key):
        return self.dic[key]
    
    def __getitem__(self,key):
        return self.dic[key]

    def __setitem__(self, key: str, value:str):
        self.dic[key]=value
            
            
        
        



                    

                

            





  

