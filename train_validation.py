# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 19:20:49 2024

@author: anuvo
"""
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(model: torch.nn.Module, loss_function: torch.nn.Module, optimizer: torch.nn.Module,
          n_epochs: int,
          training_dataloader: torch.utils.data.DataLoader,
          validation_dataloader: torch.utils.data.DataLoader,
          save_best_model: bool=False, best_model_path: str=None,
          save_fig: bool=False, fig_path: str=None,
          scheduler: torch.nn.Module=None,
          save_epoch: bool=False, checkpoint_path: str=None,
          training_loss_list: list=[], validation_loss_list: list=[], learning_rate_list: list=[],
          clip_grad: bool=False,
          detection_loss_list: list=[], identification_loss_list: list=[]):
    
    plt.ioff()
    
    best_training_loss = float('inf')
    
    with tqdm(range(n_epochs), unit=" epoch", desc="Epoch", position=0, colour='red', leave=True) as tepoch:
        tepoch.set_postfix(gpu_vram='{:.0f}MB'.format(torch.cuda.memory_allocated(DEVICE)/(1024 ** 2)),
                           gpu_util='{:.0f}%'.format(torch.cuda.utilization(DEVICE)),
                           gpu_power='{:.0f}W'.format(torch.cuda.power_draw(DEVICE)/1000))
        
        best_model = None
        
        for epoch in tepoch:
            
            training_loss_sum, validation_loss_sum = 0, 0
            
            detection_loss_sum, identification_loss_sum = 0, 0
            
            model.train()
            for i, (x, y) in enumerate(training_dataloader):
                
                optimizer.zero_grad()
                y_pred = model(x.to(DEVICE))
                loss = loss_function(y_pred, y.to(DEVICE))
                training_loss_sum += loss.item()
                
                if loss_function.stroke_identification:
                    detection_loss_sum += loss_function.temp_detection_loss.detach().item()
                    identification_loss_sum += loss_function.temp_identification_loss.detach().item()
                
                loss.backward()
                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                if len(training_loss_list)==0:                
                    tepoch.set_postfix(training='{:d}/{:d}'.format(i+1, len(training_dataloader)),
                                        training_loss='{:f}'.format(np.nan),
                                        validation_loss='{:f}'.format(np.nan),
                                        gpu_vram='{:.0f}MB'.format(torch.cuda.memory_allocated(DEVICE)/(1024 ** 2)),
                                        gpu_util='{:.0f}%'.format(torch.cuda.utilization(DEVICE)),
                                        gpu_power='{:.0f}W'.format(torch.cuda.power_draw(DEVICE)/1000),
                                        refresh=True)
                else:
                    tepoch.set_postfix(training='{:d}/{:d}'.format(i+1, len(training_dataloader)),
                                        training_loss='{:f}'.format(training_loss_list[-1]),
                                        validation_loss='{:f}'.format(validation_loss_list[-1]),
                                        gpu_vram='{:.0f}MB'.format(torch.cuda.memory_allocated(DEVICE)/(1024 ** 2)),
                                        gpu_util='{:.0f}%'.format(torch.cuda.utilization(DEVICE)),
                                        gpu_power='{:.0f}W'.format(torch.cuda.power_draw(DEVICE)/1000),
                                        refresh=True)
        
            training_loss_sum /= len(training_dataloader)
            training_loss_list.append(training_loss_sum)
            
            if loss_function.stroke_identification:
                detection_loss_sum /= len(training_dataloader)
                detection_loss_list.append(detection_loss_sum)
                identification_loss_sum/= len(training_dataloader)
                identification_loss_list.append(identification_loss_sum)
            
            model.eval()
            for i, (x, y) in enumerate(validation_dataloader):
                
                y_pred = model(x.to(DEVICE))
                loss = loss_function(y_pred, y.to(DEVICE))
                validation_loss_sum += loss.item()
                
                if len(validation_loss_list)==0:                
                    tepoch.set_postfix(validation='{:d}/{:d}'.format(i+1, len(validation_dataloader)),
                                        training_loss='{:f}'.format(training_loss_list[-1]),
                                        validation_loss='{:f}'.format(np.nan),
                                        gpu_vram='{:.0f}MB'.format(torch.cuda.memory_allocated(DEVICE)/(1024 ** 2)),
                                        gpu_util='{:.0f}%'.format(torch.cuda.utilization(DEVICE)),
                                        gpu_power='{:.0f}W'.format(torch.cuda.power_draw(DEVICE)/1000),
                                        refresh=True)
                else:
                    tepoch.set_postfix(validation='{:d}/{:d}'.format(i+1, len(validation_dataloader)),
                                        training_loss='{:f}'.format(training_loss_list[-1]),
                                        validation_loss='{:f}'.format(validation_loss_list[-1]),
                                        gpu_vram='{:.0f}MB'.format(torch.cuda.memory_allocated(DEVICE)/(1024 ** 2)),
                                        gpu_util='{:.0f}%'.format(torch.cuda.utilization(DEVICE)),
                                        gpu_power='{:.0f}W'.format(torch.cuda.power_draw(DEVICE)/1000),
                                        refresh=True)
            
            validation_loss_sum /= len(validation_dataloader)
            validation_loss_list.append(validation_loss_sum)
            
            if epoch > 1 and training_loss_list[-1] < best_training_loss:
                if not math.isnan(training_loss_list[-1]):
                    best_training_loss = training_loss_list[-1]
                    best_model = model
                
                    if save_best_model:
                        torch.save(model, best_model_path)
            
            if scheduler is not None:
                learning_rate_list.append(scheduler.get_last_lr()[0])
            #     tepoch.set_postfix(training_loss='{:f}'.format(training_loss_list[-1]),
            #                        validation_loss='{:f}'.format(validation_loss_list[-1]),
            #                        gpu_vram='{:.0f}MB'.format(torch.cuda.memory_allocated(DEVICE)/(1024 ** 2)),
            #                        gpu_util='{:.0f}%'.format(torch.cuda.utilization(DEVICE)),
            #                        gpu_power='{:.0f}W'.format(torch.cuda.power_draw(DEVICE)/1000),
            #                        learning_rate='{:f}'.format(scheduler.get_last_lr()[0]),
            #                        refresh=True)
                scheduler.step(training_loss_list[-1])
            
            else:
                learning_rate_list.append(get_lr(optimizer))
            #     tepoch.set_postfix(training_loss='{:f}'.format(training_loss_list[-1]),
            #                        validation_loss='{:f}'.format(validation_loss_list[-1]),
            #                        gpu_vram='{:.0f}MB'.format(torch.cuda.memory_allocated(DEVICE)/(1024 ** 2)),
            #                        gpu_util='{:.0f}%'.format(torch.cuda.utilization(DEVICE)),
            #                        gpu_power='{:.0f}W'.format(torch.cuda.power_draw(DEVICE)/1000),
            #                        refresh=True)
                
            if save_fig:
                fig, ax1 = plt.subplots(figsize=(10, 6))
                ax1.plot(training_loss_list, label='Training Loss', color='tab:blue', marker='o')
                ax1.plot(validation_loss_list, label='Validation Loss', color='tab:red', marker='o')
                if loss_function.stroke_identification:
                    ax1.plot(detection_loss_list, label='Training Detection Loss', color='tab:orange', marker='o')
                    ax1.plot(identification_loss_list, label='Training Identification Loss', color='tab:grey', marker='o')
                ax1.set_xlabel('Epochs')
                ax1.set_ylabel('Loss', color='black')
                ax1.tick_params(axis='y', labelcolor='black')
                ax1.legend(loc='upper left')
                # ax1.ticklabel_format(style='sci', scilimits=(0,0))

                ax2 = ax1.twinx()
                ax2.plot(learning_rate_list, label='Learning Rate', color='tab:green', linestyle='--', marker='x')
                ax2.set_ylabel('Learning Rate', color='tab:green')
                ax2.tick_params(axis='y', labelcolor='tab:green')
                ax2.legend(loc='upper right')
                # ax2.ticklabel_format(style='sci', scilimits=(0,0))
                # ax2.set_yscale('log')

                plt.title('Training and Validation Loss with Learning Rate')
                ax1.grid(True, linestyle='--', alpha=0.7)

                plt.tight_layout()
                plt.savefig(fig_path)
                plt.close(fig)
            
            if save_epoch:
                if scheduler is not None:
                    checkpoint = {'epoch': epoch,
                                  'training_loss_list': training_loss_list,
                                  'validation_loss_list': validation_loss_list,
                                  'lr_list': learning_rate_list,
                                  'model': model,
                                  'best_model': best_model,
                                  'optimizer': optimizer,
                                  'scheduler': scheduler,
                                  'detection_loss_list': detection_loss_list,
                                  'identification_loss_list': identification_loss_list}
                else:
                    checkpoint = {'epoch': epoch,
                                  'training_loss_list': training_loss_list,
                                  'validation_loss_list': validation_loss_list,
                                  'lr_list': learning_rate_list,
                                  'model': model,
                                  'best_model': best_model,
                                  'optimizer': optimizer,
                                  'detection_loss_list': detection_loss_list,
                                  'identification_loss_list': identification_loss_list}
                    
                torch.save(checkpoint, checkpoint_path)
    
    plt.ion()
    
    return training_loss_list, validation_loss_list, learning_rate_list, best_model, detection_loss_list, identification_loss_list

def evaluate_loss(model: torch.nn.Module, loss_function: torch.nn.Module,
                  training_dataloader: torch.utils.data.DataLoader,
                  validation_dataloader: torch.utils.data.DataLoader):
    
    model.eval()
    training_loss_sum, validation_loss_sum = 0, 0
    
    with tqdm(training_dataloader, unit=" sample", desc="Training", position=0, colour='red', leave=True) as t_train:
        t_train.set_postfix(gpu_vram='{:.0f}MB'.format(torch.cuda.memory_allocated(DEVICE)/(1024 ** 2)),
                           gpu_util='{:.0f}%'.format(torch.cuda.utilization(DEVICE)),
                           gpu_power='{:.0f}W'.format(torch.cuda.power_draw(DEVICE)/1000))
        
        for x, y in t_train:
            
            y_pred = model(x.to(DEVICE))
            print(y_pred, y)
            loss = loss_function(y_pred, y.to(DEVICE))
            training_loss_sum += loss.item()
            t_train.set_postfix(gpu_vram='{:.0f}MB'.format(torch.cuda.memory_allocated(DEVICE)/(1024 ** 2)),
                                gpu_util='{:.0f}%'.format(torch.cuda.utilization(DEVICE)),
                                gpu_power='{:.0f}W'.format(torch.cuda.power_draw(DEVICE)/1000),
                                refresh=True)
    
        training_loss_sum /= len(training_dataloader)
    
    with tqdm(validation_dataloader, unit=" sample", desc="Validation", position=0, colour='red', leave=True) as t_valid:
        t_train.set_postfix(gpu_vram='{:.0f}MB'.format(torch.cuda.memory_allocated(DEVICE)/(1024 ** 2)),
                           gpu_util='{:.0f}%'.format(torch.cuda.utilization(DEVICE)),
                           gpu_power='{:.0f}W'.format(torch.cuda.power_draw(DEVICE)/1000))
    
        for x, y in t_valid:
            
            y_pred = model(x.to(DEVICE))
            loss = loss_function(y_pred, y.to(DEVICE))
            validation_loss_sum += loss.item()
            
            t_valid.set_postfix(gpu_vram='{:.0f}MB'.format(torch.cuda.memory_allocated(DEVICE)/(1024 ** 2)),
                                gpu_util='{:.0f}%'.format(torch.cuda.utilization(DEVICE)),
                                gpu_power='{:.0f}W'.format(torch.cuda.power_draw(DEVICE)/1000),
                                refresh=True)
            
        validation_loss_sum /= len(validation_dataloader)

    return training_loss_sum, validation_loss_sum

def evaluate_accuracy(model: torch.nn.Module,
                      validation_dataloader: torch.utils.data.DataLoader,
                      accuracy):
    
    model.return_as_one = True
    if hasattr(model, 'stroke_logits'):
        model.stroke_logits = False
    
    model.eval()
        
    with tqdm(validation_dataloader, unit=" sample", desc="Validation", position=0, colour='red', leave=True) as t_valid:
        t_valid.set_postfix(gpu_vram='{:.0f}MB'.format(torch.cuda.memory_allocated(DEVICE)/(1024 ** 2)),
                            gpu_util='{:.0f}%'.format(torch.cuda.utilization(DEVICE)),
                            gpu_power='{:.0f}W'.format(torch.cuda.power_draw(DEVICE)/1000))
    
        for x, y in t_valid:
            
            y_pred = model(x.to(DEVICE))
            accuracy.add(y_pred, y.to(DEVICE))
            
            t_valid.set_postfix(gpu_vram='{:.0f}MB'.format(torch.cuda.memory_allocated(DEVICE)/(1024 ** 2)),
                               gpu_util='{:.0f}%'.format(torch.cuda.utilization(DEVICE)),
                               gpu_power='{:.0f}W'.format(torch.cuda.power_draw(DEVICE)/1000),
                               refresh=True)
    
    model.return_as_one = False
    if hasattr(model, 'stroke_logits'):
        model.stroke_logits = True

