# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 19:20:49 2024

@author: anuvo
"""
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, loss_function, optimizer, n_epochs, training_dataloader, validation_dataloader,
          save_model=False, model_path=None, scheduler=None):
    
    training_loss_list, validation_loss_list = [0], [0]
    learning_rate_list = []
    
    # fig, ax1 = plt.subplots(figsize=(10, 6))
    # ax1.plot(training_loss_list, label='Training Loss', color='tab:blue', marker='o')
    # ax1.plot(validation_loss_list, label='Validation Loss', color='tab:orange', marker='o')
    # ax1.set_xlabel('Epochs')
    # ax1.set_ylabel('Loss', color='black')
    # ax1.tick_params(axis='y', labelcolor='black')
    # ax1.legend(loc='upper left')
    
    # ax2 = ax1.twinx()
    # ax2.plot(learning_rate_list, label='Learning Rate', color='tab:green', linestyle='--', marker='x')
    # ax2.set_ylabel('Learning Rate', color='tab:green')
    # ax2.tick_params(axis='y', labelcolor='tab:green')
    # ax2.legend(loc='upper right')
    
    # plt.title('Training and Validation Loss with Learning Rate')
    # ax1.grid(True, linestyle='--', alpha=0.7)
    
    # plt.tight_layout()
    # plt.show()
    
    with tqdm(range(n_epochs), unit=" epoch", desc="Epoch", position=0, colour='red', leave=True) as tepoch:
        tepoch.set_postfix(gpu_vram='{:.0f}MB'.format(torch.cuda.memory_allocated(DEVICE)/(1024 ** 2)),
                           gpu_util='{:.0f}%'.format(torch.cuda.utilization(DEVICE)),
                           gpu_power='{:.0f}W'.format(torch.cuda.power_draw(DEVICE)/1000))
        
        for epoch in tepoch:
            
            training_loss_sum, validation_loss_sum = 0, 0
            
            model.train()
            for i, (x, y) in enumerate(training_dataloader):
                
                
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
        
            training_loss_sum /= len(training_dataloader)
            training_loss_list.append(training_loss_sum)
        
            
            model.eval()
            for i, (x, y) in enumerate(validation_dataloader):
                
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
            
            # ax1.plot(training_loss_list[1:], label='Training Loss', color='tab:blue', marker='o')
            # ax1.plot(validation_loss_list[1:], label='Validation Loss', color='tab:orange', marker='o')
            # ax2.plot(learning_rate_list, label='Learning Rate', color='tab:green', linestyle='--', marker='x')
            # plt.show()
    
    return training_loss_list[1:], validation_loss_list[1:], learning_rate_list

def evaluate(model, validation_dataloader, accuracy):
    
    model.return_as_one = True
    
    model.eval()
    for x, y in validation_dataloader:
        
        y_pred = model(x.to(DEVICE))
        accuracy.add(y_pred, y.to(DEVICE))

    model.return_as_one = False

