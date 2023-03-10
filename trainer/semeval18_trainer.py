
from tqdm import *
from dataloaders import *
from models import *
import torch
import math
import numpy,random
import os
import json
import pandas as pd

from datetime import datetime
from sklearn.metrics import jaccard_score,f1_score,accuracy_score
import logging



class semeval18_trainer:
    def __init__(self,opt):
        self.opt=opt
        self.device=opt.device
        self.dataloader=dataLoader(self.opt)
        self.train_loader=self.dataloader.train_loader
        self.valid_loader=self.dataloader.valid_loader
        self.test_loader=self.dataloader.test_loader
        self.model=model_classes[self.opt.model_name](self.opt).to(self.device)
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.best_valid_loss_model=None
        self.optimizer=optim_classes[self.opt.optimizer](_params,lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        self.time=datetime.strftime(datetime.now(),'%Y%m%d-%H%M%S')
        self._init_log()
        
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def _init_log(self):
        path_name=self.opt.model_name+'_'+self.opt.pretrained_bert_name+'_'+self.time
        if not os.path.exists(path_name):
            os.mkdir(path_name)
        logging.basicConfig(filename=path_name+"//train.log", filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)
        

    def run(self):
        n_epochs = self.opt.num_epoch
        best_test_acc = 0.0
        best_valid_loss=1e5


        for epoch in range(n_epochs):

            train_loss, train_acc ,train_jac,train_micro_f1,train_macro_f1= self._train()
            valid_loss, valid_acc,valid_jac,valid_micro_f1,valid_macro_f1 = self._evaluate()
            

            logging.info(f'epoch: {epoch}')
            logging.info(f'train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}, train_jac: {train_jac:.3f}, train_micro_f1: {train_micro_f1:.3f},train_macro_f1:{train_macro_f1:.3f}')
            logging.info(f'valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}, valid_jac: {valid_jac:.3f}, valid_micro_f1: {valid_micro_f1:.3f},valid_macro_f1: {valid_macro_f1:.3f}')   
                 
            if valid_loss <best_valid_loss:
                best_valid_loss = valid_loss
                self._save_model(epoch,valid_loss,valid_acc)
        print("finish train,start testing........................")
        logging.info("finish train,start testing........................")
        print("loading model: "+self.best_valid_loss_model)
        self.model.load_state_dict(torch.load(self.best_valid_loss_model,map_location=torch.device("cuda:0")))
        test_loss,test_acc,test_jac,test_micro_f1,test_macro_f1,pred,labels=self._test()
        logging.info(f'test_loss: {test_loss:.3f}, test_acc: {test_acc:.3f}, test_jac: {test_jac:.3f}, test_micro_f1: {test_micro_f1:.3f},test_macro_f1: {test_macro_f1:.3f}')
        self._save_pred_and_labels(pred,labels)
    
    def _save_model(self,epoch,loss,acc):
        path_name=self.opt.model_name+'_'+self.opt.pretrained_bert_name+'_'+self.time
        file_name="epoch_"+str(epoch)+'_'+"loss_"+'{:.4f}'.format(loss)
        if not os.path.exists(path_name):
            os.mkdir(path_name)
        with open(path_name+'//option.json','w') as f:
             json.dump(self.opt.dic,f)
        self.best_valid_loss_model=path_name+'//'+file_name+'.pt'
        torch.save(self.model.state_dict(),path_name+'//'+file_name+'.pt')
        
    def _save_pred_and_labels(self,pred,labels):
        path_name=self.opt.model_name+'_'+self.opt.pretrained_bert_name+'_'+self.time
        pred=pd.DataFrame(pred)
        labels=pd.DataFrame(labels)
        pred.to_csv(path_name+"//"+"pred.csv")
        labels.to_csv(path_name+"//"+"labels.csv")
        
            
    def _train(self):
        self.model.train()
        epoch_losses = []
        pred=[]
        labels=[]
        
        
        for batch in tqdm(self.train_loader, desc='training...'):
            
            input_ids=batch['input_ids'].to(self.device)
            token_type_ids=batch['token_type_ids'].to(self.device)
            attention_mask=batch['attention_mask'].to(self.device)
            label=batch['labels'].to(self.device)
            self.optimizer.zero_grad()
            prediction,loss = self.model(input_ids,token_type_ids,attention_mask,label)
            
            
            pred.extend((prediction>self.opt.threshold).int().cpu().numpy().flatten().tolist())
            labels.extend(label.int().cpu().numpy().flatten().tolist())
            
            loss.backward()
            self.optimizer.step()
            epoch_losses.append(loss.item())
            
        pred=np.array(pred).reshape((-1,self.opt.polarities_dim[self.opt.dataset]))
        labels=np.array(labels).reshape((-1,self.opt.polarities_dim[self.opt.dataset]))

        jaccard=jaccard_score(labels,pred,average="samples")
        micro_f1=f1_score(labels,pred,average="micro")
        macro_f1=f1_score(labels,pred,average="macro")
        accuracy=accuracy_score(labels,pred)

        return np.mean(epoch_losses), accuracy,jaccard,micro_f1,macro_f1

    def _evaluate(self):
        self.model.eval()
        epoch_losses = []
        pred=[]
        labels=[]
        
        with torch.no_grad():
            for batch in tqdm(self.valid_loader, desc='evaluating...'):
                input_ids=batch['input_ids'].to(self.device)
                token_type_ids=batch['token_type_ids'].to(self.device)
                attention_mask=batch['attention_mask'].to(self.device)
                label=batch['labels'].to(self.device)
                prediction,loss = self.model(input_ids,token_type_ids,attention_mask,label)
                
                
                pred.extend((prediction>self.opt.threshold).int().cpu().numpy().flatten().tolist())
                labels.extend(label.int().cpu().numpy().flatten().tolist())
                
                epoch_losses.append(loss.item())
                
                
        pred=np.array(pred).reshape((-1,self.opt.polarities_dim[self.opt.dataset]))
        labels=np.array(labels).reshape((-1,self.opt.polarities_dim[self.opt.dataset]))
                
        jaccard=jaccard_score(labels,pred,average="samples")
        micro_f1=f1_score(labels,pred,average="micro")
        macro_f1=f1_score(labels,pred,average="macro")
        accuracy=accuracy_score(labels,pred)

        return np.mean(epoch_losses), accuracy,jaccard,micro_f1,macro_f1

    def _test(self):
        self.model.eval()
        epoch_losses = []
        pred=[]
        labels=[]

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='testing...'):
                input_ids=batch['input_ids'].to(self.device)
                token_type_ids=batch['token_type_ids'].to(self.device)
                attention_mask=batch['attention_mask'].to(self.device)
                label=batch['labels'].to(self.device)
                prediction,loss = self.model(input_ids,token_type_ids,attention_mask,label)
                
                
                pred.extend((prediction>self.opt.threshold).int().cpu().numpy().flatten().tolist())
                labels.extend(label.int().cpu().numpy().flatten().tolist())
                
                
                epoch_losses.append(loss.item())
                
            pred=np.array(pred).reshape((-1,self.opt.polarities_dim[self.opt.dataset]))
            labels=np.array(labels).reshape((-1,self.opt.polarities_dim[self.opt.dataset]))
            
                
        jaccard=jaccard_score(labels,pred,average="samples")
        micro_f1=f1_score(labels,pred,average="micro")
        macro_f1=f1_score(labels,pred,average="macro")
        accuracy=accuracy_score(labels,pred)

        return np.mean(epoch_losses), accuracy,jaccard,micro_f1,macro_f1,pred,labels


    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel: 
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)




    