# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 19:20:49 2024

@author: anuvo
"""

import torch
from time import time

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def train(model, loss_function, optimizer, n_epochs, train_dataloader, test_dataloader): #add tqdm
    
    model.to(DEVICE)
    
    train_loss_list, test_loss_list = [], []
    
    time_start = time()
    
    for epoch in range(n_epochs):
        
        train_loss_sum, test_loss_sum = 0, 0
        
        model.train()
        for x, y in train_dataloader:
            
            optimizer.zero_grad()
            y_pred = model(x.to(DEVICE))
            loss = loss_function(y_pred, y.to(DEVICE))
            train_loss_sum += loss.item()
            loss.backward()
            optimizer.step()
        
        train_loss_sum /= len(train_dataloader)
        train_loss_list.append(train_loss_sum)
        
        model.eval()
        for x, y in test_dataloader:
            
            y_pred = model(x.to(DEVICE))
            loss = loss_function(y_pred, y.to(DEVICE))
            test_loss_sum += loss.item()
        
        test_loss_sum /= len(test_dataloader)
        test_loss_list.append(test_loss_sum)
        
        print('Epoch : {:d}/{:d} , train loss : {:.4f} , test loss : {:.4f}'.format(epoch+1, n_epochs, train_loss_sum, test_loss_sum))
        
    total_time = time() - time_start
    
    print('Total time : {:.1f}'.format(total_time))
            
    return train_loss_list, test_loss_list

def evaluate(model, test_dataloader, accuracy):
    
    model.eval()
    for x, y in test_dataloader:
        
        y_pred = model(x.to(DEVICE))
        accuracy.add(y_pred, y.to(DEVICE))


