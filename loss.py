# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 11:42:00 2024

@author: anuvo
"""

import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    y_stroke = torch.tensor([[[1] if 1 in sequence else [0] for sequence in sample] for sample in y]).type(torch.float).to(DEVICE)
    y_player = torch.tensor([[[0] if sequence[0]==1 else [1] for sequence in sample] for sample in y]).type(torch.float).to(DEVICE)
    y_hand = torch.tensor([[[0] if sequence[8]==1 else [1] for sequence in sample] for sample in y]).type(torch.float).to(DEVICE)
    y_point = torch.tensor([[point(sequence) for sequence in sample] for sample in y]).type(torch.LongTensor).to(DEVICE)
    y_serve = torch.tensor([[serve(sequence) for sequence in sample] for sample in y]).type(torch.LongTensor).to(DEVICE)

    return y_stroke, y_player, y_hand, y_point, y_serve

def adapt_1_no_void(y):
    
    y_stroke = torch.tensor([[[1] if 1 in sequence else [0] for sequence in sample] for sample in y]).type(torch.float).to(DEVICE)
    y_player = torch.tensor([[[0] if sequence[0]==1 else [1] for sequence in sample] for sample in y]).type(torch.float).to(DEVICE)
    y_hand = torch.tensor([[[0] if sequence[7]==1 else [1] for sequence in sample] for sample in y]).type(torch.float).to(DEVICE)
    y_point = torch.tensor([[point(sequence) for sequence in sample] for sample in y]).type(torch.LongTensor).to(DEVICE)
    y_serve = torch.tensor([[serve_no_void(sequence) for sequence in sample] for sample in y]).type(torch.LongTensor).to(DEVICE)

    return y_stroke, y_player, y_hand, y_point, y_serve

def adapt_2(y):
    
    y_stroke = torch.tensor([[[1] if 1 in sequence else [0] for sequence in sample] for sample in y]).type(torch.float).to(DEVICE)
    y_comb = torch.tensor([[comb(sequence) for sequence in sample] for sample in y]).type(torch.LongTensor).to(DEVICE)

    return y_stroke, y_comb

def adapt_2_no_void(y):
    
    y_stroke = torch.tensor([[[1] if 1 in sequence else [0] for sequence in sample] for sample in y]).type(torch.float).to(DEVICE)
    y_comb = torch.tensor([[comb_no_void(sequence) for sequence in sample] for sample in y]).type(torch.LongTensor).to(DEVICE)

    return y_stroke, y_comb

def bce(y, y_pred, w_0=1, w_1=1):
    return -(w_1 * y * torch.log(y_pred) + w_0 * (1-y) * torch.log(1-y_pred))

def cce(y, y_pred):
    return -torch.log(y_pred[int(y)])

class Loss_1(nn.Module):
    
    def __init__(self, void_let_serve=True, w_0: float=0.51, w_1: float=19.05):
        super(Loss_1, self).__init__()
        
        self.void_let_serve = void_let_serve
        self.w_0 = w_0
        self.w_1 = w_1
        
        # self.bce_stroke = nn.BCELoss() # default reduction == mean
        # self.bce_player = nn.BCELoss() # default reduction == mean
        # self.bce_hand = nn.BCELoss() # default reduction == mean
        # self.cce_point = nn.NLLLoss() # default reduction == mean
        # self.cce_serve = nn.NLLLoss() # default reduction == mean

    def forward(self, y_pred, y_target):
        
        y_stroke, y_player, y_hand, y_point, y_serve = adapt_1_no_void(y_target) if self.void_let_serve else adapt_1(y_target)
        y_pred_stroke, y_pred_player, y_pred_hand, y_pred_point, y_pred_serve = y_pred
        
        # y_point_pred = y_point_pred.transpose(-2, -1)
        # y_serve_pred = y_serve_pred.transpose(-2, -1)
        
        # return self.bce_stroke(y_stroke_pred, y_stroke) + y_stroke * (self.bce_player(y_player_pred, y_player) + self.bce_hand(y_hand_pred, y_hand) + self.cce_point(y_point_pred, y_point) + self.cce_serve(y_serve_pred, y_serve))
    
        batch_loss = 0
        for b in range(y_pred_stroke.shape[0]):
            sample_loss = 0
            for s in range(y_pred_stroke.shape[1]):
            
                bce_stroke = bce(y_stroke[b, s], y_pred_stroke[b, s], self.w_0, self.w_1)
                bce_player = bce(y_player[b, s], y_pred_player[b, s])
                bce_hand = bce(y_hand[b, s], y_pred_hand[b, s])
                cce_point = cce(y_point[b, s], y_pred_point[b, s])
                cce_serve = cce(y_serve[b, s], y_pred_serve[b, s])
                
                sample_loss += bce_stroke + y_stroke[b, s] * (bce_player + bce_hand + cce_point + cce_serve)

            batch_loss += sample_loss

        batch_loss /= y_pred_stroke.shape[0]*y_pred_stroke.shape[1]
        
        return batch_loss
        

class Loss_2(nn.Module):
    
    def __init__(self, void_let_serve=True, w_0: float=0.51, w_1: float=19.05):
        super(Loss_2, self).__init__()
        
        self.w_0 = w_0
        self.w_1 = w_1

        self.void_let_serve = void_let_serve
        
        # self.bce_stroke = nn.BCELoss() # default reduction == mean
        # self.cce_comb = nn.NLLLoss() # default reduction == mean
        
    def forward(self, y_pred, y):
        
        y_stroke, y_comb = adapt_2_no_void(y) if self.void_let_serve else adapt_2(y)
        y_pred_stroke, y_pred_comb = y_pred
        
        # y_comb_pred = y_comb_pred.transpose(-2, -1)
        
        # return self.bce_stroke(y_stroke_pred, y_stroke) + y_stroke * self.cce_comb(y_comb_pred, y_comb)
    
        batch_loss = 0
        for b in range(y_pred_stroke.shape[0]):
            sample_loss = 0
            for s in range(y_pred_stroke.shape[1]):
            
                bce_stroke = bce(y_stroke[b, s], y_pred_stroke[b, s], self.w_0, self.w_1)
                cce_comb = cce(y_comb[b, s], y_pred_comb[b, s])
                
                sample_loss += bce_stroke + y_stroke[b, s] * cce_comb

            batch_loss += sample_loss

        batch_loss /= y_pred_stroke.shape[0]*y_pred_stroke.shape[1]
        
        return batch_loss