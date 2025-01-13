# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 11:42:29 2024

@author: anuvo
"""

import torch
import torch.nn as nn
import numpy as np


def adapt_output_1(y_stroke, y_player, y_hand, y_point, y_serve, void_let_serve=True):
    """
    Transforms the output of Model_1 into a tensor similar to the dataset labels.

    Parameters:
    ----------
    y_stroke : torch.Tensor
        Stroke detection predictions (batch_size, seq_len, 1).
    y_player : torch.Tensor
        Player predictions (batch_size, seq_len, 1).
    y_hand : torch.Tensor
        Hand predictions (batch_size, seq_len, 1).
    y_point : torch.Tensor
        Point type predictions (batch_size, seq_len, num_classes=3).
    y_serve : torch.Tensor
        Serve type predictions (batch_size, seq_len, num_classes=4 or 5, depending on void_let_serve).

    Returns:
    -------
    torch.Tensor
        Output tensor with labels of shape (batch_size, seq_len, label_dim).
    """
    # Threshold binary predictions
    y_stroke = (y_stroke >= 0.5).long()
    y_player = (y_player >= 0.5).long()
    y_hand = (y_hand >= 0.5).long()

    # Get class indices for multi-class outputs
    y_point = torch.argmax(y_point, dim=-1)
    y_serve = torch.argmax(y_serve, dim=-1)

    # Predefine n_classes based on void_let_serve
    n_classes = 9 if void_let_serve else 10

    # Initialize the output tensor with zeros
    batch_size, seq_len = y_stroke.shape[:2]
    output = torch.zeros((batch_size, seq_len, n_classes), dtype=torch.long, device=y_stroke.device)
    
    # Create masks for stroke detection
    stroke_mask = y_stroke.squeeze(-1) == 1
    indices = torch.nonzero(stroke_mask, as_tuple=True)  # Get (batch_idx, seq_idx) for stroke_mask

    # Assign player information (Player 1 and Player 2)
    output[indices[0], indices[1], 0] = (y_player[indices[0], indices[1], 0] == 0).long()  # Player 1
    output[indices[0], indices[1], 1] = (y_player[indices[0], indices[1], 0] == 1).long()  # Player 2

    # Assign point outcomes (Point and Mistake)
    output[indices[0], indices[1], 4] = (y_point[indices[0], indices[1]] == 0).long()  # Point
    output[indices[0], indices[1], 5] = (y_point[indices[0], indices[1]] == 1).long()  # Mistake

    # Assign serve outcomes (Serve, Ball Pass, Let Serve, Void Serve)
    output[indices[0], indices[1], 2] = (y_serve[indices[0], indices[1]] == 0).long()  # Serve
    output[indices[0], indices[1], 3] = (y_serve[indices[0], indices[1]] == 1).long()  # Ball Pass
    output[indices[0], indices[1], 6] = (y_serve[indices[0], indices[1]] == 2).long()  # Let Serve
   
    # Assign hand information (Forehand and Backhand)
    if void_let_serve:
        output[indices[0], indices[1], 7] = (y_hand[indices[0], indices[1], 0] == 0).long()  # Forehand
        output[indices[0], indices[1], 8] = (y_hand[indices[0], indices[1], 0] == 1).long()  # Backhand
    else:
        output[indices[0], indices[1], 7] = (y_serve[indices[0], indices[1]] == 3).long()  # Void Serve
        output[indices[0], indices[1], 8] = (y_hand[indices[0], indices[1], 0] == 0).long()  # Forehand
        output[indices[0], indices[1], 9] = (y_hand[indices[0], indices[1], 0] == 1).long()  # Backhand

    return output


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
    y_stroke : torch.Tensor
        Stroke detection predictions (batch_size, seq_len, 1).
    y_comb : torch.Tensor
        Combined class predictions (batch_size, seq_len, num_classes).

    Returns
    -------
    torch.Tensor
        Output tensor with labels of shape (batch_size, seq_len, label_dim).
    '''
    # Binarize stroke predictions
    y_stroke = (y_stroke >= 0.5).long()

    # Get class indices for multi-class combined outputs
    y_comb = torch.argmax(y_comb, dim=-1)

    # Define the dictionary for label encoding based on `void_let_serve`
    if void_let_serve:
        label_dict = torch.tensor(list(comb_dict_no_void.values()), device=y_stroke.device)
    else:
        label_dict = torch.tensor(list(comb_dict.values()), device=y_stroke.device)

    # Initialize output tensor with zeros
    batch_size, seq_len = y_stroke.shape[:2]
    label_dim = label_dict.shape[1]
    output = torch.zeros((batch_size, seq_len, label_dim), dtype=torch.long, device=y_stroke.device)

    # Mask for detected strokes
    stroke_mask = y_stroke.squeeze(-1) == 1
    indices = torch.nonzero(stroke_mask, as_tuple=True)  # (batch_idx, seq_idx)

    # Map `y_comb` indices to label values
    comb_labels = label_dict[y_comb[indices[0], indices[1]]]

    # Assign the corresponding labels for detected strokes
    output[indices[0], indices[1]] = comb_labels

    return output
    

class Model_1(nn.Module):

    def __init__(self, sequence_len: int, n_head: int, d_model: int, num_layers: int, with_logits=False, return_as_one: bool=False, void_let_serve: bool=True):
        super(Model_1, self).__init__()
        
        self.return_as_one = return_as_one
        self.void_let_serve = void_let_serve
        self.with_logits = with_logits
        
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
        
        if self.with_logits:
            y_stroke = self.linear_stroke(x)
            y_player = self.linear_player(x)
            y_hand = self.linear_hand(x)
            y_point = self.linear_point(x)
            y_serve = self.linear_serve(x)
        else:
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
    
    
    
    
    
    
    
    
    