# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 11:42:29 2024

@author: anuvo
"""

import torch
import torch.nn as nn
import numpy as np


def adapt_output_1(y_event, y_player, y_hand, y_point, y_serve):
    
    y_event = torch.where(y_event >= 0.5, 1, 0)
    y_player = torch.where(y_player >= 0.5, 1, 0)
    y_hand = torch.where(y_hand >= 0.5, 1, 0)
    y_point = torch.argmax(y_point, dim=-1)
    y_serve = torch.argmax(y_serve, dim=-1)

    y = []
    
    if len(y_event.shape) == 3:

        for i in range(y_event.shape[0]):
            
            sample = []
            
            for j in range(y_event.shape[1]):
            
                sequence = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                
                if y_event[i, j, 0] == 0:
                    
                    sample.append(sequence)
                    
                else:
                    if y_player[i, j, 0] == 0:
                        sequence[0] = 1
                    if y_player[i, j, 0] == 1:
                        sequence[1] = 1
                        
                    if y_hand[i, j, 0] == 0:
                        sequence[8] = 1
                    if y_hand[i, j, 0] == 1:
                        sequence[9] = 1
                        
                    if y_point[i, j] == 0:
                        sequence[4] = 1
                    if y_point[i, j] == 1:
                        sequence[5] = 1
                    
                    if y_serve[i, j] == 0:
                        sequence[2] = 1
                    if y_serve[i, j] == 1:
                        sequence[3] = 1
                    if y_serve[i, j] == 2:
                        sequence[6] = 1
                    if y_serve[i, j] == 3:
                        sequence[7] = 1
                        
                    sample.append(sequence)
            
            y.append(sample)
        
        return torch.tensor(y)
    
    elif len(y_event.shape) == 2:
        
        for j in range(y_event.shape[0]):
            
            sequence = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            
            if y_event[j, 0] == 0:
            
                y.append(sequence)
            
            else:
                
                if y_player[j, 0] == 0:
                    sequence[0] = 1
                if y_player[j, 0] == 1:
                    sequence[1] = 1
                    
                if y_hand[j, 0] == 0:
                    sequence[8] = 1
                if y_hand[j, 0] == 1:
                    sequence[9] = 1
                    
                if y_point[j] == 0:
                    sequence[4] = 1
                if y_point[j] == 1:
                    sequence[5] = 1
                
                if y_serve[j] == 0:
                    sequence[2] = 1
                if y_serve[j] == 1:
                    sequence[3] = 1
                if y_serve[j] == 2:
                    sequence[6] = 1
                if y_serve[j] == 3:
                    sequence[7] = 1
                    
                y.append(sequence)
        
        return torch.tensor(y)

def adapt_output_2(y_event, y_comb):
    
    y_event = torch.where(y_event >= 0.5, 1, 0)
    y_comb = torch.argmax(y_comb, dim=-1)
    
    y = []
    
    if len(y_event.shape) == 3:

        for i in range(y_event.shape[0]):
            
            sample = []
            
            for j in range(y_event.shape[1]):
                
                if y_event[i, j, 0] == 0:
                    sample.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                
                elif y_comb[i, j] == 0:
                    sample.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
                    
                elif y_comb[i, j] == 1 :
                    sample.append([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
                    
                elif y_comb[i, j] == 2:
                    sample.append([1, 0, 1, 0, 1, 0, 0, 0, 0, 1])
                
                elif y_comb[i, j] == 3:
                    sample.append([1, 0, 1, 0, 0, 1, 0, 0, 0, 1])
                
                elif y_comb[i, j] == 4:
                    sample.append([1, 0, 0, 0, 0, 0, 1, 0, 0, 1])
                
                elif y_comb[i, j] == 5:
                    sample.append([1, 0, 0, 0, 0, 0, 0, 1, 0, 1])
                
                elif y_comb[i, j] == 6:
                    sample.append([1, 0, 0, 0, 0, 0, 0, 0, 1, 0])
                
                elif y_comb[i, j] == 7:
                    sample.append([1, 0, 1, 0, 0, 0, 0, 0, 1, 0])
                
                elif y_comb[i, j] == 8:
                    sample.append([1, 0, 1, 0, 1, 0, 0, 0, 1, 0])
                
                elif y_comb[i, j] == 9:
                    sample.append([1, 0, 1, 0, 0, 1, 0, 0, 1, 0])
                
                elif y_comb[i, j] == 10:
                    sample.append([1, 0, 0, 0, 0, 0, 1, 0, 1, 0])
                
                elif y_comb[i, j] == 11:
                    sample.append([1, 0, 0, 0, 0, 0, 0, 1, 1, 0])
                
                elif y_comb[i, j] == 12:
                    sample.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])
                
                elif y_comb[i, j] == 13:
                    sample.append([0, 1, 1, 0, 0, 0, 0, 0, 0, 1])
                
                elif y_comb[i, j] == 14:
                    sample.append([0, 1, 1, 0, 1, 0, 0, 0, 0, 1])
                
                elif y_comb[i, j] == 15:
                    sample.append([0, 1, 1, 0, 0, 1, 0, 0, 0, 1])
                
                elif y_comb[i, j] == 16:
                    sample.append([0, 1, 0, 0, 0, 0, 1, 0, 0, 1])
                
                elif y_comb[i, j] == 17:
                    sample.append([0, 1, 0, 0, 0, 0, 0, 1, 0, 1])
                
                elif y_comb[i, j] == 18:
                    sample.append([0, 1, 0, 0, 0, 0, 0, 0, 1, 0])
                
                elif y_comb[i, j] == 19:
                    sample.append([0, 1, 1, 0, 0, 0, 0, 0, 1, 0])
                
                elif y_comb[i, j] == 20:
                    sample.append([0, 1, 1, 0, 1, 0, 0, 0, 1, 0])
                
                elif y_comb[i, j] == 21:
                    sample.append([0, 1, 1, 0, 0, 1, 0, 0, 1, 0])
                
                elif y_comb[i, j] == 22:
                    sample.append([0, 1, 0, 0, 0, 0, 1, 0, 1, 0])
                
                elif y_comb[i, j] == 23:
                    sample.append([0, 1, 0, 0, 0, 0, 0, 1, 1, 0])
            
            y.append(sample)
        
        return torch.tensor(y)
    
    if len(y_event.shape) == 2:

        for j in range(y_event.shape[0]):
            
            if y_event[j, 0] == 0:
                y.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            
            elif y_comb[j] == 0:
                y.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
                
            elif y_comb[j] == 1 :
                y.append([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
                
            elif y_comb[j] == 2:
                y.append([1, 0, 1, 0, 1, 0, 0, 0, 0, 1])
            
            elif y_comb[j] == 3:
                y.append([1, 0, 1, 0, 0, 1, 0, 0, 0, 1])
            
            elif y_comb[j] == 4:
                y.append([1, 0, 0, 0, 0, 0, 1, 0, 0, 1])
            
            elif y_comb[j] == 5:
                y.append([1, 0, 0, 0, 0, 0, 0, 1, 0, 1])
            
            elif y_comb[j] == 6:
                y.append([1, 0, 0, 0, 0, 0, 0, 0, 1, 0])
            
            elif y_comb[j] == 7:
                y.append([1, 0, 1, 0, 0, 0, 0, 0, 1, 0])
            
            elif y_comb[j] == 8:
                y.append([1, 0, 1, 0, 1, 0, 0, 0, 1, 0])
            
            elif y_comb[j] == 9:
                y.append([1, 0, 1, 0, 0, 1, 0, 0, 1, 0])
            
            elif y_comb[j] == 10:
                y.append([1, 0, 0, 0, 0, 0, 1, 0, 1, 0])
            
            elif y_comb[j] == 11:
                y.append([1, 0, 0, 0, 0, 0, 0, 1, 1, 0])
            
            elif y_comb[j] == 12:
                y.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])
            
            elif y_comb[j] == 13:
                y.append([0, 1, 1, 0, 0, 0, 0, 0, 0, 1])
            
            elif y_comb[j] == 14:
                y.append([0, 1, 1, 0, 1, 0, 0, 0, 0, 1])
            
            elif y_comb[j] == 15:
                y.append([0, 1, 1, 0, 0, 1, 0, 0, 0, 1])
            
            elif y_comb[j] == 16:
                y.append([0, 1, 0, 0, 0, 0, 1, 0, 0, 1])
            
            elif y_comb[j] == 17:
                y.append([0, 1, 0, 0, 0, 0, 0, 1, 0, 1])
            
            elif y_comb[j] == 18:
                y.append([0, 1, 0, 0, 0, 0, 0, 0, 1, 0])
            
            elif y_comb[j] == 19:
                y.append([0, 1, 1, 0, 0, 0, 0, 0, 1, 0])
            
            elif y_comb[j] == 20:
                y.append([0, 1, 1, 0, 1, 0, 0, 0, 1, 0])
            
            elif y_comb[j] == 21:
                y.append([0, 1, 1, 0, 0, 1, 0, 0, 1, 0])
            
            elif y_comb[j] == 22:
                y.append([0, 1, 0, 0, 0, 0, 1, 0, 1, 0])
            
            elif y_comb[j] == 23:
                y.append([0, 1, 0, 0, 0, 0, 0, 1, 1, 0])
    
    return torch.tensor(y)
    

class Model_1(nn.Module):

    def __init__(self, sequence_len: int, return_as_one: bool=False):
        super(Model_1, self).__init__()
        
        self.return_as_one = return_as_one
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=93, nhead=31)
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.linear_event = nn.Linear(in_features=93, out_features=1)
        self.linear_player = nn.Linear(in_features=93, out_features=1)
        self.linear_hand = nn.Linear(in_features=93, out_features=1)
        self.linear_point = nn.Linear(in_features=93, out_features=3)
        self.linear_serve = nn.Linear(in_features=93, out_features=5)
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        
        x = self.transformer_encoder(x)
        y_event = self.sigmoid(self.linear_event(x))
        y_player = self.sigmoid(self.linear_player(x))
        y_hand = self.sigmoid(self.linear_hand(x))
        y_point = self.softmax(self.linear_point(x))
        y_serve = self.softmax(self.linear_serve(x))
        
        if self.return_as_one:
            
            return adapt_output_1(y_event, y_player, y_hand, y_point, y_serve)
        
        return y_event, y_player, y_hand, y_point, y_serve
    
class Model_2(nn.Module):

    def __init__(self, sequence_len: int, return_as_one: bool=False):
        super(Model_2, self).__init__()

        self.return_as_one = return_as_one

        encoder_layer = nn.TransformerEncoderLayer(d_model=93, nhead=31)
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.linear_event = nn.Linear(in_features=93, out_features=1)
        self.linear_comb = nn.Linear(in_features=93, out_features=24)
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        
        x = self.transformer_encoder(x)
        y_event = self.sigmoid(self.linear_event(x))
        y_comb = self.softmax(self.linear_comb(x))
        
        if self.return_as_one:
            
            return adapt_output_2(y_event, y_comb)
        
        return y_event, y_comb
    
    
    
    
    
    
    
    
    