# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 11:42:00 2024

@author: anuvo
"""

import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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

combination_sequence_map = torch.tensor([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    [1, 0, 1, 0, 1, 0, 0, 0, 1, 0],
    [1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 0, 0, 1],
    [0, 1, 1, 0, 0, 1, 0, 0, 0, 1],
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 0, 1, 0, 0, 0, 1, 0],
    [0, 1, 1, 0, 0, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
], dtype=torch.float32, device=DEVICE)

# Vectorize the function using torch.vmap
def get_class_comb(sample):
    
    comparisons = (sample.unsqueeze(1) == combination_sequence_map).all(dim=2).int()
    indices = torch.argmax(comparisons, dim=1)
    
    return indices

comb_no_void_sequence_map = torch.tensor([
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 0, 1, 0, 0, 0, 0, 1, 0],
    [1, 0, 1, 0, 1, 0, 0, 1, 0],
    [1, 0, 1, 0, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1],
    [0, 1, 1, 0, 0, 0, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 0, 1],
    [0, 1, 1, 0, 0, 1, 0, 0, 1],
    [0, 1, 0, 0, 0, 0, 1, 0, 1],
    [0, 1, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 0, 1, 0, 0, 1, 0],
    [0, 1, 1, 0, 0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 1, 0],
], dtype=torch.float32, device=DEVICE)

def get_class_comb_no_void(sample):
    """
    Maps a sample of input sequences to corresponding indices based on a predefined sequence-to-index mapping.
    
    Args:
    sample (torch.Tensor): A tensor of shape `(sequence_length, 9)` representing the input 
                          sequences to be mapped. Each sequence is expected to match one of the 
                          predefined sequences in `comb_no_void_sequence_map`.
    
    Returns:
    torch.Tensor: A tensor of shape `(sequence_length,)` containing the indices of the matched sequences 
                  in `comb_no_void_sequence_map`. Each element corresponds to the index of the first 
                  matching sequence for the respective input sequence in the sample.
  
    Example:
    >>> sample = torch.tensor([
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 1, 1, 0]
        ])
    >>> get_class_comb_no_void(sample)
    tensor([0, 6, 19])  # Indices of the matched sequences
    """
    comparisons = (sample.unsqueeze(1) == comb_no_void_sequence_map).all(dim=2).int()
    indices = torch.argmax(comparisons, dim=1)
    
    return indices
    
def adapt_target_to_output_1(y):

    # Check for strokes
    y_stroke = (y.sum(dim=-1, keepdim=True) > 0).float()
       
    # Determine player class (1 if first element is 0, else 0)
    y_player = (y[:, :, 0] == 0).float().unsqueeze(-1)
       
    # Determine hand class (1 if feature at index 8 is 0, else 0)
    y_hand = (y[:, :, 8] == 0).float().unsqueeze(-1)
       
    # Compute point class (vectorized mapping using tensor indexing)
    y_point = torch.where(
        y[:, :, 4] == 1, 0,  # If index 4 is 1, point class is 0
        torch.where(y[:, :, 5] == 1, 1, 2)  # If index 5 is 1, point class is 1, else 2
    )
       
    # Compute serve class (vectorized mapping using tensor indexing)
    y_serve = torch.where(
        y[:, :, 2] == 1, 0,  # If index 2 is 1, serve class is 0
        torch.where(
            y[:, :, 3] == 1, 1,  # If index 3 is 1, serve class is 1
            torch.where(
                y[:, :, 6] == 1, 2,   # If index 6 is 1, serve class is 2
                torch.where(
                    y[:, :, 7] == 1, 3, 4  # If index 7 is 1, serve class is 3, else 4
                )
            )
        )
    )

    return y_stroke, y_player, y_hand, y_point.long(), y_serve.long()

def adapt_target_to_output_1_no_void(y):
    
    # Check for strokes
    y_stroke = (y.sum(dim=-1, keepdim=True) > 0).float()
       
    # Determine player class (1 if first element is 0, else 0)
    y_player = (y[:, :, 0] == 0).float().unsqueeze(-1)
       
    # Determine hand class (1 if feature at index 7 is 0, else 0)
    y_hand = (y[:, :, 7] == 0).float().unsqueeze(-1)
       
    # Compute point class (vectorized mapping using tensor indexing)
    y_point = torch.where(
        y[:, :, 4] == 1, 0,  # If index 4 is 1, point class is 0
        torch.where(y[:, :, 5] == 1, 1, 2)  # If index 5 is 1, point class is 1, else 2
    )
       
    # Compute serve class (vectorized mapping using tensor indexing)
    y_serve = torch.where(
        y[:, :, 2] == 1, 0,  # If index 2 is 1, serve class is 0
        torch.where(
            y[:, :, 3] == 1, 1,  # If index 3 is 1, serve class is 1
            torch.where(
                y[:, :, 6] == 1, 2, 3  # If index 6 is 1, serve class is 2, else 3
            )
        )
    )

    return y_stroke, y_player, y_hand, y_point.long(), y_serve.long()

def adapt_target_to_output_2(y):
    
    y_stroke = (y.sum(dim=-1, keepdim=True) > 0).float()
    y_comb = torch.vmap(get_class_comb)(y)

    return y_stroke, y_comb

def adapt_target_to_output_2_no_void(y):
    """
    """
    
    y_stroke = (y.sum(dim=-1, keepdim=True) > 0).float()
    y_comb = torch.vmap(get_class_comb_no_void)(y)

    return y_stroke, y_comb

def bce(y_pred, y_target, w_0=1, w_1=1):
    # Apply BCE loss for all elements in the tensor
    return -(w_1 * y_target * torch.log(y_pred) + w_0 * (1 - y_target) * torch.log(1 - y_pred))

def cce(y_pred, y_target):
    # Extract the predicted probability for the correct class index for each sample
    batch_size, num_samples, num_classes = y_pred.shape
    
    # Convert the class indices to the correct locations in y_pred using advanced indexing
    y_pred_correct_class = y_pred[torch.arange(batch_size).unsqueeze(1), torch.arange(num_samples), y_target.int()]
    
    # Calculate the categorical cross-entropy for each sample
    return -torch.log(y_pred_correct_class)


class Loss_1(nn.Module):
    
    def __init__(self, void_let_serve=True, w_0: float=1, w_1: float=1, stroke_identification=True):
        super(Loss_1, self).__init__()
        
        self.void_let_serve = void_let_serve
        self.w_0 = w_0
        self.w_1 = w_1
        self.stroke_identification = stroke_identification
        
        self.temp_detection_loss = 0
        self.temp_identification_loss = 0

    def forward(self, y_pred, y_target):
        
        self.temp_detection_loss = 0
        self.temp_identification_loss = 0
        
        y_stroke, y_player, y_hand, y_point, y_serve = adapt_target_to_output_1_no_void(y_target) if self.void_let_serve else adapt_target_to_output_1(y_target)
        y_pred_stroke, y_pred_player, y_pred_hand, y_pred_point, y_pred_serve = y_pred
        
        bce_stroke = bce(y_pred_stroke, y_stroke, self.w_0, self.w_1)
        
        if self.stroke_identification :
            bce_player = bce(y_pred_player, y_player)
            bce_hand = bce(y_pred_hand, y_hand)
            
            cce_point = cce(y_pred_point, y_point)
            cce_serve = cce(y_pred_serve, y_serve)
        
            sample_loss = bce_stroke + y_stroke * (bce_player + bce_hand + cce_point.unsqueeze(-1) + cce_serve.unsqueeze(-1))
            
            self.temp_identification_loss = sample_loss.detach() - bce_stroke.detach()
            self.temp_identification_loss = self.temp_identification_loss.sum() / (y_pred_stroke.shape[0] * y_pred_stroke.shape[1])
            
        else:
            sample_loss = bce_stroke
    
        # Sum over all batches and samples to get the total batch loss
        batch_loss = sample_loss.sum() / (y_pred_stroke.shape[0] * y_pred_stroke.shape[1])
        
        self.temp_detection_loss = bce_stroke.detach().sum() / (y_pred_stroke.shape[0] * y_pred_stroke.shape[1])
    
        return batch_loss


class Loss_1_with_logits(nn.Module):
    
    def __init__(self, void_let_serve=True, pos_weight: float=1, stroke_identification=True):
        super(Loss_1_with_logits, self).__init__()
        
        self.void_let_serve = void_let_serve
        self.stroke_identification = stroke_identification
        
        self.temp_detection_loss = 0
        self.temp_identification_loss = 0
        
        self.bce_stroke = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([pos_weight]).to(DEVICE))
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.cce = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, y_pred, y_target):
        
        self.temp_detection_loss = 0
        self.temp_identification_loss = 0
        
        y_stroke, y_player, y_hand, y_point, y_serve = adapt_target_to_output_1_no_void(y_target) if self.void_let_serve else adapt_target_to_output_1(y_target)
        y_pred_stroke, y_pred_player, y_pred_hand, y_pred_point, y_pred_serve = y_pred
        
        bce_stroke = self.bce_stroke(y_pred_stroke, y_stroke)
        
        if self.stroke_identification :
            bce_player = self.bce(y_pred_player, y_player)
            bce_hand = self.bce(y_pred_hand, y_hand)

            cce_point = self.cce(y_pred_point.permute(0, 2, 1), y_point)
            cce_serve = self.cce(y_pred_serve.permute(0, 2, 1), y_serve)
        
            sample_loss = bce_stroke + y_stroke * (bce_player + bce_hand + cce_point.unsqueeze(-1) + cce_serve.unsqueeze(-1))
            
            self.temp_identification_loss = sample_loss.detach() - bce_stroke.detach()
            self.temp_identification_loss = self.temp_identification_loss.sum() / (y_pred_stroke.shape[0] * y_pred_stroke.shape[1])
            
        else:
            sample_loss = bce_stroke
    
        # Sum over all batches and samples to get the total batch loss
        batch_loss = sample_loss.sum() / (y_pred_stroke.shape[0] * y_pred_stroke.shape[1])
        
        self.temp_detection_loss = bce_stroke.detach().sum() / (y_pred_stroke.shape[0] * y_pred_stroke.shape[1])

        return batch_loss
        

class Loss_2(nn.Module):
    
    def __init__(self, void_let_serve=True, w_0: float=1, w_1: float=1, stroke_identification=True):
        super(Loss_2, self).__init__()
        
        self.w_0 = w_0
        self.w_1 = w_1

        self.void_let_serve = void_let_serve
        self.stroke_identification = stroke_identification
        
        self.temp_detection_loss = 0
        self.temp_identification_loss = 0
        
    def forward(self, y_pred, y_target):
        
        self.temp_detection_loss = 0
        self.temp_identification_loss = 0
        
        y_stroke, y_comb = adapt_target_to_output_2_no_void(y_target) if self.void_let_serve else adapt_target_to_output_2(y_target)
        y_pred_stroke, y_pred_comb = y_pred

        bce_stroke = bce(y_pred_stroke, y_stroke, self.w_0, self.w_1)

        if self.stroke_identification :
            # Compute the categorical cross-entropy for the combination predictions
            cce_comb = cce(y_pred_comb, y_comb)
            # Now, element-wise sum of BCE and CCE for each sample
            sample_loss = bce_stroke + y_stroke * cce_comb.unsqueeze(-1)
            
            self.temp_identification_loss = sample_loss.detach() - bce_stroke.detach()
            self.temp_identification_loss = self.temp_identification_loss.sum() / (y_pred_stroke.shape[0] * y_pred_stroke.shape[1])
        
        else:
            sample_loss = bce_stroke
    
        # Sum over all batches and samples to get the total batch loss
        batch_loss = sample_loss.sum() / (y_pred_stroke.shape[0] * y_pred_stroke.shape[1])
        
        self.temp_detection_loss = bce_stroke.detach().sum() / (y_pred_stroke.shape[0] * y_pred_stroke.shape[1])
        
        return batch_loss


class Loss_2_with_logits(nn.Module):
    
    def __init__(self, void_let_serve=True, pos_weight: float=1, stroke_identification=True):
        super(Loss_2_with_logits, self).__init__()
        
        self.void_let_serve = void_let_serve
        self.stroke_identification = stroke_identification
        
        self.temp_detection_loss = 0
        self.temp_identification_loss = 0
        
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([pos_weight]).to(DEVICE))
        self.cce = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, y_pred, y_target):
        
        self.temp_detection_loss = 0
        self.temp_identification_loss = 0
        
        y_stroke, y_comb = adapt_target_to_output_2_no_void(y_target) if self.void_let_serve else adapt_target_to_output_2(y_target)
        y_pred_stroke, y_pred_comb = y_pred
        
        bce_stroke = self.bce(y_pred_stroke, y_stroke)
        
        if self.stroke_identification :
            cce_comb = self.cce(y_pred_comb.permute(0, 2, 1), y_comb)
        
            sample_loss = bce_stroke + y_stroke * cce_comb.unsqueeze(-1)
            
            self.temp_identification_loss = sample_loss.detach() - bce_stroke.detach()
            self.temp_identification_loss = self.temp_identification_loss.sum() / (y_pred_stroke.shape[0] * y_pred_stroke.shape[1])
            
        else:
            sample_loss = bce_stroke
    
        # Sum over all batches and samples to get the total batch loss
        batch_loss = sample_loss.sum() / (y_pred_stroke.shape[0] * y_pred_stroke.shape[1])
        
        self.temp_detection_loss = bce_stroke.detach().sum() / (y_pred_stroke.shape[0] * y_pred_stroke.shape[1])

        return batch_loss