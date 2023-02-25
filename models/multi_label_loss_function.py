from cv2 import threshold
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial,reduce
from torch.nn import BCEWithLogitsLoss

class multi_label_loss_base(nn.Module):

    def __init__(self,opt, in_features, out_features):

        super().__init__()
        self.opt=opt
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        if opt.dataset=="goemotions":
            self.corr=np.load("./data/goemotions_data/corr_matrix.npy")
        elif opt.dataset=="semeval18":
            self.corr=np.load("./data/semeval18_data/corr_matrix.npy")
        


    def forward(self, x, labels):

        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features
        
        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)
          
        return wf,self.s_circle_loss(wf,labels)
    
    def uni_loss(self,pred,label):
        s=self.opt.s
        m=self.opt.m
        threshold=self.opt.threshold

        pred_pos=-s*(pred)-(1-label)*1e12
        pred_neg=s*(pred)-label*1e12
        
        ones=s*torch.ones((pred.shape[0],1)).to(pred.device)
        pred_neg=torch.exp(torch.cat([pred_neg,threshold*ones],axis=-1))
        pred_pos=torch.exp(torch.cat([pred_pos,-threshold*ones],axis=-1))
        
        pair_matrix=torch.einsum("ij,ik->ijk",pred_pos,pred_neg)
        loss=torch.log(1+torch.einsum("ijk->i",pair_matrix))

        loss=loss.sum(axis=0)/loss.numel()


        return loss
        
        
    def s_circle_loss(self,pred,label):
        
        s=self.opt.s
        m=self.opt.m
        threshold=self.opt.threshold
        _soft_label=self.corr if self.opt.dataset in ["goemotions","semeval18"] else np.zeros((self.opt.polarities_dim[self.opt.dataset],self.opt.polarities_dim[self.opt.dataset]))
        _soft_label=torch.tensor(_soft_label).float().to(pred.device)
        soft_label=torch.sigmoid(torch.tanh(_soft_label*3)).detach()
        
        
        _neg=torch.einsum("ij,ik->ikj",pred,torch.ones_like(pred))
        _pos=torch.einsum("ij,ik->ijk",pred,torch.ones_like(pred))
        

        alpha_neg=torch.sigmoid(_neg+threshold+soft_label*0.2).detach()
        alpha_pos=torch.sigmoid(-(_pos-threshold)).detach()
        
        neg=(_neg+threshold+soft_label*0.2)*alpha_neg
        pos=(_pos-threshold)*alpha_pos

    

        l_matrix=torch.einsum("ij,ik->ijk",label,1-label)
        mask=(1-l_matrix)*1e12
        
        matrix=s*(neg-pos)-mask
        
        loss1=torch.einsum("ijk->i",torch.exp(matrix))
        

        pos2=s*torch.sigmoid((pred-threshold)*2)*(pred-threshold)-label*1e12
        neg2=s*torch.sigmoid((threshold-pred)*2)*(threshold-pred)-(1-label)*1e12
        loss2=torch.exp(pos2).sum(axis=-1)+torch.exp(neg2).sum(axis=-1)
        
        loss=torch.log(1+loss1+loss2)
        
        loss=loss.sum(axis=0)/loss.numel()

        return loss


    def circle_loss(self,pred,label):
            s=self.opt.s
            m=self.opt.m
            threshold=self.opt.threshold
            
            op=1+m
            on=-1-m
            delta_p=1-m
            delta_n=-1+m
            
            _neg=torch.einsum("ij,ik->ikj",pred,torch.ones_like(pred))
            _pos=torch.einsum("ij,ik->ijk",pred,torch.ones_like(pred))
            
            alpha_neg=_neg-delta_n
            alpha_pos=delta_p-_neg
            
            neg=(_neg-delta_n)*alpha_neg
            pos=(_pos-delta_p)*alpha_pos

            l_matrix=torch.einsum("ij,ik->ijk",label,1-label)
            mask=(1-l_matrix)*1e12
            
            matrix=s*(neg-pos)-mask
            
            loss1=torch.einsum("ijk->i",torch.exp(matrix))
            
            loss=loss.sum(axis=0)/loss.numel()
            
            return loss
        
    
        
    

