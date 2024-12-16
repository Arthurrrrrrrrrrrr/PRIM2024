# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 19:20:49 2024

@author: anuvo
"""
import time
import torch
import numpy as np
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, loss_function, optimizer, n_epochs, training_dataloader, validation_dataloader,
          save_model=False, model_path=None, scheduler=None):
    
    training_loss_list, validation_loss_list = [0], [0]
    learning_rate_list = []
    
    benchmark_data = [[[], []], [[], []], [[], []], [[], []]]
    
    with tqdm(range(n_epochs), unit=" epoch", desc="Epoch", position=0, colour='red', leave=True) as tepoch:
        tepoch.set_postfix(gpu_vram='{:.0f}MB'.format(torch.cuda.memory_allocated(DEVICE)/(1024 ** 2)),
                           gpu_util='{:.0f}%'.format(torch.cuda.utilization(DEVICE)),
                           gpu_power='{:.0f}W'.format(torch.cuda.power_draw(DEVICE)/1000))
        
        for epoch in tepoch:
            
            training_loss_sum, validation_loss_sum = 0, 0
            
            model.train()
            end = time.time()
            for i, (x, y) in enumerate(training_dataloader):
                
                benchmark_data[0][0].append(time.time()-end)
                
                optimizer.zero_grad()
                y_pred = model(x.to(DEVICE))
                loss = loss_function(y_pred, y.to(DEVICE))
                training_loss_sum += loss.item()
                loss.backward()
                optimizer.step()
                
                tepoch.set_postfix(training='{:d}/{:d}'.format(i+1, len(training_dataloader)),
                                    training_loss='{:f}'.format(training_loss_list[-1]),
                                    validation_loss='{:f}'.format(validation_loss_list[-1]),
                                    gpu_vram='{:.0f}MB'.format(torch.cuda.memory_allocated(DEVICE)/(1024 ** 2)),
                                    gpu_util='{:.0f}%'.format(torch.cuda.utilization(DEVICE)),
                                    gpu_power='{:.0f}W'.format(torch.cuda.power_draw(DEVICE)/1000),
                                    refresh=True)
                
                benchmark_data[1][0].append(time.time()-end)
                end = time.time()
                benchmark_data[2][0].append(torch.cuda.utilization(DEVICE))
                benchmark_data[3][0].append(torch.cuda.memory_allocated(DEVICE)/(1024 ** 2))
        
            training_loss_sum /= len(training_dataloader)
            training_loss_list.append(training_loss_sum)
        
            
            model.eval()
            end = time.time()
            for i, (x, y) in enumerate(validation_dataloader):
                
                benchmark_data[0][1].append(time.time()-end)
                
                y_pred = model(x.to(DEVICE))
                loss = loss_function(y_pred, y.to(DEVICE))
                validation_loss_sum += loss.item()
                
                tepoch.set_postfix(validation='{:d}/{:d}'.format(i+1, len(validation_dataloader)),
                                    training_loss='{:f}'.format(training_loss_list[-1]),
                                    validation_loss='{:f}'.format(validation_loss_list[-1]),
                                    gpu_vram='{:.0f}MB'.format(torch.cuda.memory_allocated(DEVICE)/(1024 ** 2)),
                                    gpu_util='{:.0f}%'.format(torch.cuda.utilization(DEVICE)),
                                    gpu_power='{:.0f}W'.format(torch.cuda.power_draw(DEVICE)/1000),
                                    refresh=True)
                
                benchmark_data[1][1].append(time.time()-end)
                end = time.time()
                benchmark_data[2][1].append(torch.cuda.utilization(DEVICE))
                benchmark_data[3][1].append(torch.cuda.memory_allocated(DEVICE)/(1024 ** 2))
                
            validation_loss_sum /= len(validation_dataloader)
            validation_loss_list.append(validation_loss_sum)
            
            if epoch >= 2 and save_model and training_loss_list[-1]<training_loss_list[-2]:
                
                torch.save(model, model_path)
                
            if scheduler is not None:
                learning_rate_list.append(scheduler.get_last_lr()[0])
                tepoch.set_postfix(training_loss='{:f}'.format(training_loss_sum),
                                   validation_loss='{:f}'.format(validation_loss_sum),
                                   gpu_vram='{:.0f}MB'.format(torch.cuda.memory_allocated(DEVICE)/(1024 ** 2)),
                                   gpu_util='{:.0f}%'.format(torch.cuda.utilization(DEVICE)),
                                   gpu_power='{:.0f}W'.format(torch.cuda.power_draw(DEVICE)/1000),
                                   learning_rate='{:f}'.format(scheduler.get_last_lr()[0]),
                                   refresh=True)
                scheduler.step(training_loss_list[-1])
            
            else:
                tepoch.set_postfix(training_loss='{:f}'.format(training_loss_sum),
                                   validation_loss='{:f}'.format(validation_loss_sum),
                                   gpu_vram='{:.0f}MB'.format(torch.cuda.memory_allocated(DEVICE)/(1024 ** 2)),
                                   gpu_util='{:.0f}%'.format(torch.cuda.utilization(DEVICE)),
                                   gpu_power='{:.0f}W'.format(torch.cuda.power_draw(DEVICE)/1000),
                                   refresh=True) 
                
    
    return training_loss_list[1:], validation_loss_list[1:], learning_rate_list, benchmark_data

def evaluate(model, validation_dataloader, accuracy):
    
    model.eval()
    for x, y in validation_dataloader:
        
        y_pred = model(x.to(DEVICE))
        accuracy.add(y_pred, y.to(DEVICE))


