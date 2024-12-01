# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 19:00:45 2024

@author: anuvo
"""
import numpy as np

class AccuracyTest():
    
    def __init__(self, sequence_len, adapt: callable=None, void_let_serve=True):
        
        self.sequence_len = sequence_len
        self.adapt = adapt
        self.void_let_serve = void_let_serve
        
        self.event_detection_accuracy = np.zeros((4)) # FN, TP, TN, FP
        
        if self.void_let_serve:
            self.event_identification_accuracy = np.zeros((4, self.sequence_len, 9)) # FN, TP, TN, FP
        else:    
            self.event_identification_accuracy = np.zeros((4, self.sequence_len, 10)) # FN, TP, TN, FP
        
    def add(self, y_preds, ys):

        if self.adapt is not None:
            y_preds = self.adapt(*y_preds)
      
        if len(y_preds.shape) == 3:

            for i in range(y_preds.shape[0]):
                
                y_pred, y = y_preds[i], ys[i]      
            
                for t in range(self.sequence_len):
                    
                    if 1 in y[t]:
                        if not 1 in y_pred[t]:
                            self.event_detection_accuracy[0] += 1
                        else:
                            self.event_detection_accuracy[1] += 1
                    else:
                        if not 1 in y_pred[t]:
                            self.event_detection_accuracy[2] += 1
                        else:
                            self.event_detection_accuracy[3] += 1
                    
                    if 1 in y[t]:
                        fn = [is_fn(y[t, k], y_pred[t, k]) for k in range(y_preds.shape[2])]
                        tp = [is_tp(y[t, k], y_pred[t, k]) for k in range(y_preds.shape[2])]
                        tn = [is_tn(y[t, k], y_pred[t, k]) for k in range(y_preds.shape[2])]
                        fp = [is_fp(y[t, k], y_pred[t, k]) for k in range(y_preds.shape[2])]
                        
                        self.event_identification_accuracy[0, t] += fn
                        self.event_identification_accuracy[1, t] += tp
                        self.event_identification_accuracy[2, t] += tn
                        self.event_identification_accuracy[3, t] += fp
                
        elif len(y_preds.shape) == 2:
            
            for t in range(self.sequence_len):
                
                if 1 in y[t]:
                    if not 1 in y_pred[t]:
                        self.event_detection_accuracy[0] += 1
                    else:
                        self.event_detection_accuracy[1] += 1
                else:
                    if not 1 in y_pred[t]:
                        self.event_detection_accuracy[2] += 1
                    else:
                        self.event_detection_accuracy[3] += 1
                
                if 1 in y[t]:
                    fn = [is_fn(y[t, k], y_pred[t, k]) for k in range(y_preds.shape[1])]
                    tp = [is_tp(y[t, k], y_pred[t, k]) for k in range(y_preds.shape[1])]
                    tn = [is_tn(y[t, k], y_pred[t, k]) for k in range(y_preds.shape[1])]
                    fp = [is_fp(y[t, k], y_pred[t, k]) for k in range(y_preds.shape[1])]
                    
                    self.event_identification_accuracy[0, t] += fn
                    self.event_identification_accuracy[1, t] += tp
                    self.event_identification_accuracy[2, t] += tn
                    self.event_identification_accuracy[3, t] += fp

        
    def metrics(self):
        
        return self.event_detection_accuracy, self.event_identification_accuracy

        
def is_fn(y, y_pred):
    
    if y == 1 and y_pred == 0:
        return 1
    else:
        return 0
   
def is_tp(y, y_pred):
    
    if y == 1 and y_pred == 1:
        return 1
    else:
        return 0
   
def is_tn(y, y_pred):
    
    if y == 0 and y_pred == 0:
        return 1
    else:
        return 0
   
def is_fp(y, y_pred):
    
    if y == 0 and y_pred == 1:
        return 1
    else:
        return 0
        
        
        
        
        
        
        