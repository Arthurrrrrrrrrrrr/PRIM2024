# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 11:42:29 2024

@author: anuvo
"""

import torch
import torch.nn as nn
import numpy as np


def adapt_output_1(y_stroke, y_player, y_hand, y_point, y_serve, void_let_serve=True):
    '''
    Transforms the output of Model_1 into a tensor similar to the dataset labels.

    Parameters
    ----------
    y_stroke : TYPE
        DESCRIPTION.
    y_player : TYPE
        DESCRIPTION.
    y_hand : TYPE
        DESCRIPTION.
    y_point : TYPE
        DESCRIPTION.
    y_serve : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    y_stroke = torch.where(y_stroke >= 0.5, 1, 0)
    y_player = torch.where(y_player >= 0.5, 1, 0)
    y_hand = torch.where(y_hand >= 0.5, 1, 0)
    y_point = torch.argmax(y_point, dim=-1)
    y_serve = torch.argmax(y_serve, dim=-1)

    y = []
    
    if void_let_serve:
        n = 1
    else:
        n = 0

    for i in range(y_stroke.shape[0]):
        
        sample = []
        
        for j in range(y_stroke.shape[1]):
            
            if void_let_serve:
                sequence = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            else:
                sequence = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            
            if y_stroke[i, j, 0] == 0: # no stroke detected
                sample.append(sequence)
                
            else: # stroke detected and corresponding strokes identified
                if y_player[i, j, 0] == 0: #player1
                    sequence[0] = 1
                if y_player[i, j, 0] == 1: #player2
                    sequence[1] = 1
                
                if y_hand[i, j, 0] == 0: #forehand
                    sequence[8-n] = 1
                if y_hand[i, j, 0] == 1: #backhand
                    sequence[9-n] = 1
                    
                if y_point[i, j] == 0: #point
                    sequence[4] = 1
                if y_point[i, j] == 1: #mistake
                    sequence[5] = 1
                                       #None
                if y_serve[i, j] == 0: #serve
                    sequence[2] = 1
                if y_serve[i, j] == 1: #ball pass
                    sequence[3] = 1
                
                if y_serve[i, j] == 2: #let serve (or void_serve if mixed)
                        sequence[6] = 1
                if not void_let_serve:
                    if y_serve[i, j] == 3: #void serve
                        sequence[7] = 1
                                       #None
                sample.append(sequence)
        
        y.append(sample)
    
    return torch.tensor(y)

comb_dict = {0: [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             1: [1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
             2: [1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
             3: [1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
             4: [1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
             5: [1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
             6: [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
             7: [1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
             8: [1, 0, 1, 0, 1, 0, 0, 0, 1, 0],
             9: [1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
             10: [1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
             11: [1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
             12: [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
             13: [0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
             14: [0, 1, 1, 0, 1, 0, 0, 0, 0, 1],
             15: [0, 1, 1, 0, 0, 1, 0, 0, 0, 1],
             16: [0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
             17: [0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
             18: [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
             19: [0, 1, 1, 0, 0, 0, 0, 0, 1, 0],
             20: [0, 1, 1, 0, 1, 0, 0, 0, 1, 0],
             21: [0, 1, 1, 0, 0, 1, 0, 0, 1, 0],
             22: [0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
             23: [0, 1, 0, 0, 0, 0, 0, 1, 1, 0]}

comb_dict_no_void = {0: [1, 0, 0, 0, 0, 0, 0, 0, 1],
                     1: [1, 0, 1, 0, 0, 0, 0, 0, 1],
                     2: [1, 0, 1, 0, 1, 0, 0, 0, 1],
                     3: [1, 0, 1, 0, 0, 1, 0, 0, 1],
                     4: [1, 0, 0, 0, 0, 0, 1, 0, 1],
                     5: [1, 0, 0, 0, 0, 0, 0, 1, 0],
                     6: [1, 0, 1, 0, 0, 0, 0, 1, 0],
                     7: [1, 0, 1, 0, 1, 0, 0, 1, 0],
                     8: [1, 0, 1, 0, 0, 1, 0, 1, 0],
                     9: [1, 0, 0, 0, 0, 0, 1, 1, 0],
                     10: [0, 1, 0, 0, 0, 0, 0, 0, 1],
                     11: [0, 1, 1, 0, 0, 0, 0, 0, 1],
                     12: [0, 1, 1, 0, 1, 0, 0, 0, 1],
                     13: [0, 1, 1, 0, 0, 1, 0, 0, 1],
                     14: [0, 1, 0, 0, 0, 0, 1, 0, 1],
                     15: [0, 1, 0, 0, 0, 0, 0, 1, 0],
                     16: [0, 1, 1, 0, 0, 0, 0, 1, 0],
                     17: [0, 1, 1, 0, 1, 0, 0, 1, 0],
                     18: [0, 1, 1, 0, 0, 1, 0, 1, 0],
                     19: [0, 1, 0, 0, 0, 0, 1, 1, 0]}

def adapt_output_2(y_stroke, y_comb, void_let_serve=True):
    '''
    Transforms the output of Model_2 into a tensor similar to the dataset labels.

    Parameters
    ----------
    y_stroke : TYPE
        DESCRIPTION.
    y_comb : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    y_stroke = torch.where(y_stroke >= 0.5, 1, 0)
    y_comb = torch.argmax(y_comb, dim=-1)
    
    y = []
    
    for i in range(y_stroke.shape[0]):
        
        sample = []
        
        for j in range(y_stroke.shape[1]):
            
            if y_stroke[i, j, 0] == 0: #no stroke detected
                if void_let_serve:
                    sample.append([0, 0, 0, 0, 0, 0, 0, 0, 0])
                else:
                    sample.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) 
        
            else:  # stroke detected and corresponding strokes identified
                if void_let_serve:
                    sample.append(comb_dict_no_void[y_comb[i, j].item()])
                else:
                    sample.append(comb_dict[y_comb[i, j]])
        
        y.append(sample)
    
    return torch.tensor(y)
    

class Model_1(nn.Module):

    def __init__(self, sequence_len: int, n_head: int, d_model: int, num_layers: int, return_as_one: bool=False, void_let_serve: bool=True):
        super(Model_1, self).__init__()
        
        self.return_as_one = return_as_one
        self.void_let_serve = void_let_serve
        
        self.linear_embedding = nn.Linear(in_features=93, out_features=d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.linear_stroke = nn.Linear(in_features=d_model, out_features=1)
        self.linear_player = nn.Linear(in_features=d_model, out_features=1)
        self.linear_hand = nn.Linear(in_features=d_model, out_features=1)
        self.linear_point = nn.Linear(in_features=d_model, out_features=3)
        
        if self.void_let_serve:
            self.linear_serve = nn.Linear(in_features=d_model, out_features=4)
        else:
            self.linear_serve = nn.Linear(in_features=d_model, out_features=5)
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        
        x = self.linear_embedding(x)
        x = self.transformer_encoder(x)
        y_stroke = self.sigmoid(self.linear_stroke(x))
        y_player = self.sigmoid(self.linear_player(x))
        y_hand = self.sigmoid(self.linear_hand(x))
        y_point = self.softmax(self.linear_point(x))
        y_serve = self.softmax(self.linear_serve(x))
        
        if self.return_as_one:
            
            return adapt_output_1(y_stroke, y_player, y_hand, y_point, y_serve, self.void_let_serve)
        
        return y_stroke, y_player, y_hand, y_point, y_serve
    
class Model_2(nn.Module):

    def __init__(self, sequence_len: int, n_head: int, d_model: int, num_layers: int, return_as_one: bool=False, void_let_serve: bool=True):
        super(Model_2, self).__init__()

        self.return_as_one = return_as_one
        self.void_let_serve = void_let_serve
        
        self.linear_embedding = nn.Linear(in_features=93, out_features=d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.linear_stroke = nn.Linear(in_features=d_model, out_features=1)
        
        if self.void_let_serve:
            self.linear_comb = nn.Linear(in_features=d_model, out_features=20)
        else:
            self.linear_comb = nn.Linear(in_features=d_model, out_features=24)
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        
        x = self.linear_embedding(x)
        x = self.transformer_encoder(x)
        y_stroke = self.sigmoid(self.linear_stroke(x))
        y_comb = self.softmax(self.linear_comb(x))
        
        if self.return_as_one:
            
            return adapt_output_2(y_stroke, y_comb, self.void_let_serve)
        
        return y_stroke, y_comb
    
    
    
    
    
    
    
    
    