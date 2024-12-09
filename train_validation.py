# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 19:20:49 2024

@author: anuvo
"""
import torch
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, loss_function, optimizer, n_epochs, train_dataloader, validation_dataloader):
    
    train_loss_list, validation_loss_list = [], []
    
    with tqdm(range(n_epochs), unit=" epoch", desc="Epoch", position=0, colour='blue', leave=True) as tepoch:
        tepoch.set_postfix(gpu_vram='{:.0f}MB'.format(torch.cuda.memory_allocated(DEVICE)/(1024 ** 2)),
                           gpu_util='{:.0f}%'.format(torch.cuda.utilization(DEVICE)),
                           gpu_power='{:.0f}W'.format(torch.cuda.power_draw(DEVICE)/1000))
        
        for epoch in tepoch:
            
            train_loss_sum, validation_loss_sum = 0, 0
            
            model.train()
            with tqdm(train_dataloader, unit=" batch", desc="Train", position=1, colour='white', leave=False) as ttrain:
                for x, y in ttrain:
                    
                    optimizer.zero_grad()
                    y_pred = model(x.to(DEVICE))
                    loss = loss_function(y_pred, y.to(DEVICE))
                    train_loss_sum += loss.item()
                    loss.backward()
                    optimizer.step()
                    
                    # ttrain.set_postfix(train_loss='{:f}'.format(train_loss_sum), refresh=True)
            
                train_loss_sum /= len(train_dataloader)
                train_loss_list.append(train_loss_sum)
        
            
            model.eval()
            with tqdm(validation_dataloader, unit=" batch",  desc="Validation", position=2, colour='red', leave=False) as tvalid:
                for x, y in tvalid:
                
                    y_pred = model(x.to(DEVICE))
                    loss = loss_function(y_pred, y.to(DEVICE))
                    validation_loss_sum += loss.item()
                    
                    # tvalid.set_postfix(validation_loss='{:f}'.format(validation_loss_sum), refresh=True)
                    
                validation_loss_sum /= len(validation_dataloader)
                validation_loss_list.append(validation_loss_sum)
            
            tepoch.set_postfix(train_loss='{:f}'.format(train_loss_sum),
                               validation_loss='{:f}'.format(validation_loss_sum),
                               gpu_vram='{:.0f}MB'.format(torch.cuda.memory_allocated(DEVICE)/(1024 ** 2)),
                               gpu_util='{:.0f}%'.format(torch.cuda.utilization(DEVICE)),
                               gpu_power='{:.0f}W'.format(torch.cuda.power_draw(DEVICE)/1000),
                               refresh=True)
    
    
    return train_loss_list, validation_loss_list

def evaluate(model, validation_dataloader, accuracy):
    
    model.eval()
    for x, y in validation_dataloader:
        
        y_pred = model(x.to(DEVICE))
        accuracy.add(y_pred, y.to(DEVICE))


