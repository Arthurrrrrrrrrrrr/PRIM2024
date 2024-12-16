# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 19:00:45 2024

@author: anuvo
"""
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AccuracyTest():
    
    def __init__(self, sequence_len, void_let_serve=True):
        
        self.sequence_len = sequence_len
        self.void_let_serve = void_let_serve
        
        self.stroke_detection_accuracy = torch.zeros((4)).to(DEVICE) # FN, TP, TN, FP
        
        if self.void_let_serve:
            self.stroke_identification_accuracy = torch.zeros((4, self.sequence_len, 9)).to(DEVICE) # FN, TP, TN, FP
        else:    
            self.stroke_identification_accuracy = torch.zeros((4, self.sequence_len, 10)).to(DEVICE) # FN, TP, TN, FP
        
    def add(self, y_pred, y):
    
        # Binary stroke detection: presence of any stroke in a time step
        stroke_present = (y.sum(axis=-1) > 0).int() # Shape: (batch_size, seq_len)
        pred_stroke_present = (y_pred.sum(axis=-1) > 0).int()  # Shape: (batch_size, seq_len)
    
        # Stroke detection accuracy: FN, TP, TN, FP
        self.stroke_detection_accuracy[0] += torch.sum((stroke_present == 1) & (pred_stroke_present == 0))  # FN
        self.stroke_detection_accuracy[1] += torch.sum((stroke_present == 1) & (pred_stroke_present == 1))  # TP
        self.stroke_detection_accuracy[2] += torch.sum((stroke_present == 0) & (pred_stroke_present == 0))  # TN
        self.stroke_detection_accuracy[3] += torch.sum((stroke_present == 0) & (pred_stroke_present == 1))  # FP
    
        # Stroke identification accuracy metrics
        fn = ((y == 1) & (y_pred == 0))  # FN
        tp = ((y == 1) & (y_pred == 1))  # TP
        tn = ((y == 0) & (y_pred == 0))  # TN
        fp = ((y == 0) & (y_pred == 1))  # FP
    
    
        # Update metrics for stroke identification
        self.stroke_identification_accuracy[0] += fn.sum(axis=0)  # FN
        self.stroke_identification_accuracy[1] += tp.sum(axis=0)  # TP
        self.stroke_identification_accuracy[2] += tn.sum(axis=0)  # TN
        self.stroke_identification_accuracy[3] += fp.sum(axis=0)  # FP
        
    def metrics(self):
        
        return self.stroke_detection_accuracy, self.stroke_identification_accuracy


        
        
        
        
        
        
        