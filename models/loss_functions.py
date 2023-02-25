import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial,reduce



class AngularPenaltySMLoss(nn.Module):

    def __init__(self, in_features, out_features, loss_type='arcface', eps=1e-7, s=None, m=None):
        '''
        Angular Penalty Softmax Loss

        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers: 
        
        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599

        '''
        super().__init__()
        loss_type = loss_type.lower()
        assert loss_type in  ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features
        
        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)
        

        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)

        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)) + self.m)

        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)))


        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        
        L = numerator - torch.log(denominator)
    
        
        return wf,-torch.mean(L)
    
    def init_w(self):
        w=torch.zeros_like(self.fc.weight)
        w[0,0]=1
        w[-1]=-w[0]
        w[2,1]=1
        w[1]=F.normalize(w[0]+w[2],dim=0)
        w[3]=F.normalize(w[-1]+w[2],dim=0)
        self.fc.weight=torch.nn.Parameter(w,requires_grad=False)

    
    
class AngularPenaltySMLossWithSoftLabel(AngularPenaltySMLoss):
    def __init__(self, in_features, out_features, loss_type='arcface', eps=1e-7, s=None, m=None,k=None):
        super().__init__(in_features,out_features,loss_type,eps, s, m)
        self.k=k
        
    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features
        
        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)
        
        soft_onehot=self.cal_soft_one_hot(labels,self.out_features,self.out_features,self.k)
        
        soft_onehot=soft_onehot.cuda()

        if self.loss_type == 'cosface':
            
            margins=torch.zeros_like(wf).scatter_(1,labels.unsqueeze(1),self.m)
            
            numerators=self.s*(wf-margins)
            
            numerator =torch.diagonal(wf.transpose(0, 1)[labels])

        if self.loss_type == 'arcface':
            
            margins=torch.zeros_like(wf).scatter_(1,labels.unsqueeze(1),self.m)
            
            numerators=self.s * torch.cos(torch.acos(torch.clamp(wf, -1.+self.eps, 1-self.eps)) + self.m)
            
            numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)) + self.m)

        if self.loss_type == 'sphereface':
            
            margins=torch.ones_like(wf).scatter_(1,labels.unsqueeze(1),self.m)
            
            numerators=self.s*torch.cos(margins * torch.acos(torch.clamp(wf, -1.+self.eps, 1-self.eps)))
            
            numerator = self.s * torch.cos(self.m * torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)))

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        
        L1= numerators - torch.log(denominator).unsqueeze(1).repeat((1,self.out_features))
        
        L=torch.einsum('ij,ij->i',soft_onehot,L1)
        
        return wf,-torch.mean(L)
    
    def cal_soft_one_hot(self,labels,label_dim,window_size,k):
        
        labels=labels.to('cpu')
        
        batch_size=labels.shape[0]
        
        onehot=torch.zeros((batch_size,label_dim)).scatter_(1,labels.unsqueeze(1),1).unsqueeze(0).unsqueeze(0)
        
        pad_onehot=torch.nn.functional.pad(onehot,(window_size,window_size,0,0)).squeeze(0).squeeze(0).unsqueeze(-1).transpose(1,2)
        
        conv=torch.nn.Conv1d(1,1,kernel_size=window_size,bias=False)
        
        kernel=torch.tensor([np.exp(-np.abs(window_size//2-i)*k) for i in range(0,window_size)]).unsqueeze(0).unsqueeze(0).float()

        conv.weight.data=kernel
        
        start=(window_size+1)//2
        
        numerator=conv(pad_onehot)[:,:,start:start+label_dim].squeeze(1)

        denominator=torch.einsum("ij->i",numerator).unsqueeze(1).repeat((1,label_dim))
        
        return torch.div(numerator,denominator)
        
        
        
        
class MultiAngularPenaltySMLoss(AngularPenaltySMLoss):

    def __init__(self, in_features, out_features, loss_type='arcface', eps=1e-7, s=None, m=None,em_num_of_perclass=3,inner_margin=0.2):
        '''
        Angular Penalty Softmax Loss

        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers: 
        
        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599

        '''
        super().__init__(in_features,out_features,loss_type,eps, s, m)

        self.em_num_of_perclass=em_num_of_perclass
        self.inner_margin=inner_margin
        self.fc = nn.Sequential(
            *[nn.Linear(in_features, out_features, bias=False) for i in range(self.em_num_of_perclass)]
            )

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features
        
        
        for f in self.fc:
            for W in f.parameters():
                W = F.normalize(W, p=2, dim=1)
                
                
        x = F.normalize(x, p=2, dim=1)
                
        l=[]        
        for i in range(self.em_num_of_perclass):
            l.append(self.fc[i](x))
        

        wf,_= torch.max(torch.stack(l,dim=-1),dim=-1)
        

        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)

        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)) + self.m)

        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)))


        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        
        L = numerator - torch.log(denominator)
    
        
        return wf,-torch.mean(L)+self.inner_loss()
    
    def inner_loss(self):
        loss=0
        for i in range(self.em_num_of_perclass):
            l=[list(j.parameters())[0][:,i] for j in self.fc]
            w=torch.stack(l,dim=0)
            loss+=self._inner_loss(w,self.inner_margin)
        return loss
            
    
    def _inner_loss(self,w,margin):
        wt=w.T
        R=torch.clamp(torch.einsum('ij,jk->ik',w,wt),-1.0+self.eps,1.0-self.eps)
        arc_length=torch.acos(R)
        loss_matrix=torch.triu(arc_length,diagonal=1)
        loss=torch.einsum("ij->",loss_matrix)/loss_matrix.numel()
        
        return loss
        

        
