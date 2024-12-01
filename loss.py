# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 11:42:00 2024

@author: anuvo
"""

import torch
import torch.nn as nn

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


bce = nn.BCEWithLogitsLoss() #default reduction == mean
cce = nn.CrossEntropyLoss() # same


def point(sequence):
    
    if sequence[4]==1: 
        return 0
    elif sequence[5]==1:
        return 1
    else:
        return 2
    
def serve(sequence):
    
    if sequence[2]==1: 
        return 0
    elif sequence[3]==1:
        return 1
    elif sequence[6]==1:
        return 2
    elif sequence[7]==1:
        return 3
    else:
        return 4
    
def serve_no_void(sequence):
    
    if sequence[2]==1: 
        return 0
    elif sequence[3]==1:
        return 1
    elif sequence[6]==1:
        return 2
    else:
        return 3
    
def comb(sequence):
    
    if sequence == [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]:
        return 0
    elif sequence == [1, 0, 1, 0, 0, 0, 0, 0, 0, 1]:
        return 1
    elif sequence == [1, 0, 1, 0, 1, 0, 0, 0, 0, 1]:
        return 2
    elif sequence == [1, 0, 1, 0, 0, 1, 0, 0, 0, 1]:
        return 3
    elif sequence == [1, 0, 0, 0, 0, 0, 1, 0, 0, 1]:
        return 4
    elif sequence == [1, 0, 0, 0, 0, 0, 0, 1, 0, 1]:
        return 5
    elif sequence == [1, 0, 0, 0, 0, 0, 0, 0, 1, 0]:
        return 6
    elif sequence == [1, 0, 1, 0, 0, 0, 0, 0, 1, 0]:
        return 7
    elif sequence == [1, 0, 1, 0, 1, 0, 0, 0, 1, 0]:
        return 8
    elif sequence == [1, 0, 1, 0, 0, 1, 0, 0, 1, 0]:
        return 9
    elif sequence == [1, 0, 0, 0, 0, 0, 1, 0, 1, 0]:
        return 10
    elif sequence == [1, 0, 0, 0, 0, 0, 0, 1, 1, 0]:
        return 11
    elif sequence == [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]:
        return 12
    elif sequence == [0, 1, 1, 0, 0, 0, 0, 0, 0, 1]:
        return 13
    elif sequence == [0, 1, 1, 0, 1, 0, 0, 0, 0, 1]:
        return 14
    elif sequence == [0, 1, 1, 0, 0, 1, 0, 0, 0, 1]:
        return 15
    elif sequence == [0, 1, 0, 0, 0, 0, 1, 0, 0, 1]:
        return 16
    elif sequence == [0, 1, 0, 0, 0, 0, 0, 1, 0, 1]:
        return 17
    elif sequence == [0, 1, 0, 0, 0, 0, 0, 0, 1, 0]:
        return 18
    elif sequence == [0, 1, 1, 0, 0, 0, 0, 0, 1, 0]:
        return 19
    elif sequence == [0, 1, 1, 0, 1, 0, 0, 0, 1, 0]:
        return 20
    elif sequence == [0, 1, 1, 0, 0, 1, 0, 0, 1, 0]:
        return 21
    elif sequence == [0, 1, 0, 0, 0, 0, 1, 0, 1, 0]:
        return 22
    elif sequence == [0, 1, 0, 0, 0, 0, 0, 1, 1, 0]:
        return 23
    else:
        return 0
    
def comb_no_void(sequence):
    
    if sequence == [1, 0, 0, 0, 0, 0, 0, 0, 1]:
        return 0
    elif sequence == [1, 0, 1, 0, 0, 0, 0, 0, 1]:
        return 1
    elif sequence == [1, 0, 1, 0, 1, 0, 0, 0, 1]:
        return 2
    elif sequence == [1, 0, 1, 0, 0, 1, 0, 0, 1]:
        return 3
    elif sequence == [1, 0, 0, 0, 0, 0, 1, 0, 1]:
        return 4
    elif sequence == [1, 0, 0, 0, 0, 0, 0, 1, 0]:
        return 5
    elif sequence == [1, 0, 1, 0, 0, 0, 0, 1, 0]:
        return 6
    elif sequence == [1, 0, 1, 0, 1, 0, 0, 1, 0]:
        return 7
    elif sequence == [1, 0, 1, 0, 0, 1, 0, 1, 0]:
        return 8
    elif sequence == [1, 0, 0, 0, 0, 0, 1, 1, 0]:
        return 9
    elif sequence == [0, 1, 0, 0, 0, 0, 0, 0, 1]:
        return 10
    elif sequence == [0, 1, 1, 0, 0, 0, 0, 0, 1]:
        return 11
    elif sequence == [0, 1, 1, 0, 1, 0, 0, 0, 1]:
        return 12
    elif sequence == [0, 1, 1, 0, 0, 1, 0, 0, 1]:
        return 13
    elif sequence == [0, 1, 0, 0, 0, 0, 1, 0, 1]:
        return 14
    elif sequence == [0, 1, 0, 0, 0, 0, 0, 1, 0]:
        return 15
    elif sequence == [0, 1, 1, 0, 0, 0, 0, 1, 0]:
        return 16
    elif sequence == [0, 1, 1, 0, 1, 0, 0, 1, 0]:
        return 17
    elif sequence == [0, 1, 1, 0, 0, 1, 0, 1, 0]:
        return 18
    elif sequence == [0, 1, 0, 0, 0, 0, 1, 1, 0]:
        return 19
    else:
        return 0
    
def adapt_1(y):
    
    y_event = torch.tensor([[[1] if 1 in sequence else [0] for sequence in sample] for sample in y]).type(torch.float).to(DEVICE)
    y_player = torch.tensor([[[0] if sequence[0]==1 else [1] for sequence in sample] for sample in y]).type(torch.float).to(DEVICE)
    y_hand = torch.tensor([[[0] if sequence[8]==1 else [1] for sequence in sample] for sample in y]).type(torch.float).to(DEVICE)
    y_point = torch.tensor([[point(sequence) for sequence in sample] for sample in y]).type(torch.LongTensor).to(DEVICE)
    y_serve = torch.tensor([[serve(sequence) for sequence in sample] for sample in y]).type(torch.LongTensor).to(DEVICE)

    return y_event, y_player, y_hand, y_point, y_serve

def adapt_1_no_void(y):
    
    y_event = torch.tensor([[[1] if 1 in sequence else [0] for sequence in sample] for sample in y]).type(torch.float).to(DEVICE)
    y_player = torch.tensor([[[0] if sequence[0]==1 else [1] for sequence in sample] for sample in y]).type(torch.float).to(DEVICE)
    y_hand = torch.tensor([[[0] if sequence[7]==1 else [1] for sequence in sample] for sample in y]).type(torch.float).to(DEVICE)
    y_point = torch.tensor([[point(sequence) for sequence in sample] for sample in y]).type(torch.LongTensor).to(DEVICE)
    y_serve = torch.tensor([[serve_no_void(sequence) for sequence in sample] for sample in y]).type(torch.LongTensor).to(DEVICE)

    return y_event, y_player, y_hand, y_point, y_serve

def adapt_2(y):
    
    y_event = torch.tensor([[[1] if 1 in sequence else [0] for sequence in sample] for sample in y]).type(torch.float).to(DEVICE)
    y_comb = torch.tensor([[comb(sequence) for sequence in sample] for sample in y]).type(torch.LongTensor).to(DEVICE)

    return y_event, y_comb

def adapt_2_no_void(y):
    
    y_event = torch.tensor([[[1] if 1 in sequence else [0] for sequence in sample] for sample in y]).type(torch.float).to(DEVICE)
    y_comb = torch.tensor([[comb_no_void(sequence) for sequence in sample] for sample in y]).type(torch.LongTensor).to(DEVICE)

    return y_event, y_comb

class Loss_1(nn.Module):
    
    def __init__(self, void_let_serve=True):
        super(Loss_1, self).__init__()
        
        self.void_let_serve = void_let_serve

    def forward(self, y_pred, y):
        
        y_event, y_player, y_hand, y_point, y_serve = adapt_1_no_void(y) if self.void_let_serve else adapt_1(y)
        y_event_pred, y_player_pred, y_hand_pred, y_point_pred, y_serve_pred = y_pred
        
        y_point_pred = y_point_pred.transpose(-2, -1)
        y_serve_pred = y_serve_pred.transpose(-2, -1)
        
        return bce(y_event_pred, y_event) + bce(y_player_pred, y_player) + bce(y_hand_pred, y_hand) + cce(y_point_pred, y_point) + cce(y_serve_pred, y_serve)
    

class Loss_2(nn.Module):
    
    def __init__(self, void_let_serve=True):
        super(Loss_2, self).__init__()

        self.void_let_serve = void_let_serve
        
    def forward(self, y_pred, y):
        
        y_event, y_comb = adapt_2_no_void(y) if self.void_let_serve else adapt_2(y)
        y_event_pred, y_comb_pred = y_pred
        
        y_comb_pred = y_comb_pred.transpose(-2, -1)
        
        return bce(y_event_pred, y_event) + cce(y_comb_pred, y_comb)
    
    