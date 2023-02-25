from genericpath import exists
from tqdm import *
from dataloaders import *
from models import *
import torch
import math
import numpy,random
import os
import json
from draw import *

from sklearn.metrics import confusion_matrix 

from datetime import datetime



class trainer:
    def __init__(self,opt):
        self.opt=opt
        self.device=opt.device
        self.dataloader=dataLoader(self.opt)
        self.train_loader=self.dataloader.train_loader
        self.valid_loader=self.dataloader.valid_loader
        self.test_loader=self.dataloader.test_loader
        self.model=model_classes[self.opt.model_name](self.opt).to(self.device)
        self.criterion=torch.nn.CrossEntropyLoss().to(self.device)
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer=optim_classes[self.opt.optimizer](_params,lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        self.time=datetime.strftime(datetime.now(),'%Y%m%d-%H%M%S')
        self.path=self.opt.model_name+'_'+self.opt.pretrained_bert_name+'_'+self.time
        
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def run(self):
        n_epochs = self.opt.num_epoch
        best_test_acc = 0.0
        

        for epoch in range(n_epochs):

            train_loss, train_acc = self._train()
            valid_loss, valid_acc = self._evaluate()
            test_loss,test_acc,pred,labels=self._test()


            epoch_train_loss = np.mean(train_loss)
            epoch_train_acc = np.mean(train_acc)
            epoch_valid_loss = np.mean(valid_loss)
            epoch_valid_acc = np.mean(valid_acc)
            epoch_test_loss = np.mean(test_loss)
            epoch_test_acc = np.mean(test_acc)
            
            if epoch_test_acc > best_test_acc:
                best_test_acc = epoch_test_acc
                self._save_model(self.opt,epoch,epoch_test_loss,epoch_test_acc)
                self._save_con_m(pred,labels,epoch)
                

            print(f'epoch: {epoch+1}')
            print(f'train_loss: {epoch_train_loss:.3f}, train_acc: {epoch_train_acc:.3f}')
            print(f'valid_loss: {epoch_valid_loss:.3f}, valid_acc: {epoch_valid_acc:.3f}')        
            print(f'test_loss: {epoch_test_loss:.3f}, test_acc: {epoch_test_acc:.3f}')
            
    def _save_con_m(self,pred,labels,epoch):
        cm=confusion_matrix(labels,np.argmax(pred,axis=1))
        plot_confusion_matrix(cm,[0,1,2,3,4],self.path+f'//confusion_matrix_epoch_{epoch}.png')
    
    
    def _save_model(self,opt,epoch,loss,acc):
        path_name=self.path
        file_name="epoch_"+str(epoch)+'_'+"acc_"+'{:.4f}'.format(acc)
        if not os.path.exists(path_name):
            os.mkdir(path_name)
            with open(path_name+'//option.json','w') as f:
                json.dump(self.opt.dic,f)
            
        torch.save(self.model.state_dict(),path_name+'//'+file_name+'.pt')
            
    def _train(self):
        self.model.train()
        epoch_losses = []
        epoch_accs = []
        
        for batch in tqdm(self.train_loader, desc='training...'):
            input_ids=batch['input_ids'].to(self.device)
            token_type_ids=batch['token_type_ids'].to(self.device)
            attention_mask=batch['attention_mask'].to(self.device)
            label=batch['labels'].to(self.device)
            self.optimizer.zero_grad()
            prediction,loss = self.model(input_ids,token_type_ids,attention_mask,label)
            if loss==None:
                loss = self.criterion(prediction, label)
            accuracy = self.get_accuracy(prediction, label)
            loss.backward()
            self.optimizer.step()
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())

        return epoch_losses, epoch_accs

    def _evaluate(self):
        self.model.eval()
        epoch_losses = []
        epoch_accs = []

        with torch.no_grad():
            for batch in tqdm(self.valid_loader, desc='evaluating...'):
                input_ids=batch['input_ids'].to(self.device)
                token_type_ids=batch['token_type_ids'].to(self.device)
                attention_mask=batch['attention_mask'].to(self.device)
                label=batch['labels'].to(self.device)
                prediction,loss = self.model(input_ids,token_type_ids,attention_mask,label)
                if loss==None:
                    loss = self.criterion(prediction, label)
                accuracy = self.get_accuracy(prediction, label)
                epoch_losses.append(loss.item())
                epoch_accs.append(accuracy.item())

        return epoch_losses, epoch_accs

    def _test(self):
        self.model.eval()
        epoch_losses = []
        epoch_accs = []
        pred=[]
        labels=[]

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='testing...'):
                input_ids=batch['input_ids'].to(self.device)
                token_type_ids=batch['token_type_ids'].to(self.device)
                attention_mask=batch['attention_mask'].to(self.device)
                label=batch['labels'].to(self.device)
                prediction,loss = self.model(input_ids,token_type_ids,attention_mask,label)
                if loss==None:
                    loss = self.criterion(prediction, label)
                pred.extend(prediction.cpu().numpy().flatten().tolist())
                labels.extend(label.cpu().numpy().flatten().tolist())
                accuracy = self.get_accuracy(prediction, label)
                epoch_losses.append(loss.item())
                epoch_accs.append(accuracy.item())
                
        pred=np.array(pred).reshape(-1,self.opt.polarities_dim[self.opt.dataset])
        labels=np.array(labels)

        return epoch_losses, epoch_accs,pred,labels


    def get_accuracy(self,prediction, label):
        batch_size, _ = prediction.shape
        predicted_classes = prediction.argmax(dim=-1)
        correct_predictions = predicted_classes.eq(label).sum()
        accuracy = correct_predictions / batch_size
        return accuracy


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




    