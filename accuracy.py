# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 19:00:45 2024

@author: anuvo
"""
import torch
import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Accuracy():
    
    def __init__(self, sequence_len, void_let_serve=True):
        
        self.metrics_computed = False
        
        self.sequence_len = sequence_len
        self.void_let_serve = void_let_serve
        
        self.simple_accuracy = torch.zeros((2, self.sequence_len)).to(DEVICE) # Correct, Wrong
        
        self.stroke_detection_accuracy = torch.zeros((4, self.sequence_len)).to(DEVICE) # FN, FP, TN, TP
        
        if self.void_let_serve:
            self.stroke_identification_accuracy = torch.zeros((4, self.sequence_len, 9)).to(DEVICE) # FN, FP, TN, TP
        else:    
            self.stroke_identification_accuracy = torch.zeros((4, self.sequence_len, 10)).to(DEVICE) # FN, FP, TN, TP
            
        self.player_mismatch = torch.zeros((4)).to(DEVICE) # player 1 instead of 2, player 2 instead of 1, player 1 count, player 2 count
        self.hand_mismatch = torch.zeros((4)).to(DEVICE) # backhand instead of forehand, forehand instead of backhand, backhand count, forehand count
            
        if self.void_let_serve:
            self.labels_map = {
                0: "player 1",
                1: "player 2",
                2: "serve",
                3: "ball pass",
                4: "point",
                5: "mistake",
                6: "let serve or void serve",
                7: "forehand",
                8: "backhand"}
        else: 
            self.labels_map = {
                0: "player 1",
                1: "player 2",
                2: "serve",
                3: "ball pass",
                4: "point",
                5: "mistake",
                6: "let serve",
                7: "void serve",
                8: "forehand",
                9: "backhand"}
        
    def add(self, y_pred, y_target):
    
        # Binary stroke detection: presence of any stroke in a time step
        stroke_present = (y_target.sum(axis=-1) > 0).int() # Shape: (batch_size, seq_len)
        pred_stroke_present = (y_pred.sum(axis=-1) > 0).int()  # Shape: (batch_size, seq_len)

        # Simple accuracy: Correct, Wrong
        self.simple_accuracy[0] += torch.all(torch.eq(y_target, y_pred), dim=2).sum(axis=0)
        self.simple_accuracy[1] += (~torch.all(torch.eq(y_target, y_pred), dim=2)).sum(axis=0)
        
        # Stroke detection accuracy: FN, TP, TN, FP
        self.stroke_detection_accuracy[0] += ((stroke_present == 1) & (pred_stroke_present == 0)).sum(axis=0)  # FN
        self.stroke_detection_accuracy[1] += ((stroke_present == 0) & (pred_stroke_present == 1)).sum(axis=0)  # FP
        self.stroke_detection_accuracy[2] += ((stroke_present == 0) & (pred_stroke_present == 0)).sum(axis=0)  # TN
        self.stroke_detection_accuracy[3] += ((stroke_present == 1) & (pred_stroke_present == 1)).sum(axis=0)  # TP
    
        # Stroke identification accuracy metrics
        fn = ((y_target == 1) & (y_pred == 0))  # FN Shape: (batch_size, seq_len, nb_classes)
        fp = ((y_target == 0) & (y_pred == 1))  # FP Shape: (batch_size, seq_len, nb_classes)
        tn = ((y_target == 0) & (y_pred == 0))  # TN Shape: (batch_size, seq_len, nb_classes)
        tp = ((y_target == 1) & (y_pred == 1))  # TP Shape: (batch_size, seq_len, nb_classes)

        # Update metrics for stroke identification
        self.stroke_identification_accuracy[0] += fn.sum(axis=0)  # FN
        self.stroke_identification_accuracy[1] += fp.sum(axis=0)  # FP
        self.stroke_identification_accuracy[2] += tn.sum(axis=0)  # TN
        self.stroke_identification_accuracy[3] += tp.sum(axis=0)  # TP
        
        # Player mismatch
        self.player_mismatch[0] += ((y_pred[:, :, 0] == 1) & (y_target[:, :, 1] == 1)).sum() # player 1 instead of 2
        self.player_mismatch[1] += ((y_pred[:, :, 1] == 1) & (y_target[:, :, 0] == 1)).sum() # player 2 instead of 1
        self.player_mismatch[2] += (y_target[:, :, 0] == 1).sum() # player 1
        self.player_mismatch[3] += (y_target[:, :, 1] == 1).sum() # player 2
        
        # Hand mismatch
        self.hand_mismatch[0] += ((y_pred[:, :, -1] == 1) & (y_target[:, :, -2] == 1)).sum() # backhand instead of forehand
        self.hand_mismatch[1] += ((y_pred[:, :, -2] == 1) & (y_target[:, :, -1] == 1)).sum() # forehand instead of backhand
        self.hand_mismatch[2] += (y_target[:, :, -1] == 1).sum() # backhand count
        self.hand_mismatch[3] += (y_target[:, :, -2] == 1).sum() # forehand count

    def compute_metrics(self):
        
        epsilon = 1e-8
        
        self.global_simple_accuracy = self.simple_accuracy[0].sum()/(self.simple_accuracy[0].sum()+self.simple_accuracy[1].sum()+epsilon) # Shape: (1)
        self.by_frame_simple_accuracy = self.simple_accuracy[0]/(self.simple_accuracy[0]+self.simple_accuracy[1]+epsilon) # Shape: (seq_len)
        
        self.global_detection_precision = self.stroke_detection_accuracy[3].sum()/(self.stroke_detection_accuracy[3].sum()+self.stroke_detection_accuracy[1].sum()+epsilon) # Shape: (1)
        self.global_detection_recall = self.stroke_detection_accuracy[3].sum()/(self.stroke_detection_accuracy[3].sum()+self.stroke_detection_accuracy[0].sum()+epsilon) # Shape: (1)
        
        self.by_frame_detection_precision = self.stroke_detection_accuracy[3]/(self.stroke_detection_accuracy[3]+self.stroke_detection_accuracy[1]+epsilon) # Shape: (seq_len)
        self.by_frame_detection_recall = self.stroke_detection_accuracy[3]/(self.stroke_detection_accuracy[3]+self.stroke_detection_accuracy[0]+epsilon) # Shape: (seq_len)
        
        self.global_identification_precision = self.stroke_identification_accuracy[3].sum(axis=0)/(self.stroke_identification_accuracy[3].sum(axis=0)+self.stroke_identification_accuracy[1].sum(axis=0)+epsilon) # Shape: (nb_classes)
        self.global_identification_recall = self.stroke_identification_accuracy[3].sum(axis=0)/(self.stroke_identification_accuracy[3].sum(axis=0)+self.stroke_identification_accuracy[0].sum(axis=0)+epsilon) # Shape: (nb_classes)
        
        self.by_frame_identification_precision = self.stroke_identification_accuracy[3]/(self.stroke_identification_accuracy[3]+self.stroke_identification_accuracy[1]+epsilon) # Shape: (seq_len, nb_classes)
        self.by_frame_identification_recall = self.stroke_identification_accuracy[3]/(self.stroke_identification_accuracy[3]+self.stroke_identification_accuracy[0]+epsilon) # Shape: (seq_len, nb_classes)
        
        self.metrics_computed = True
    
    def display_metrics(self):
        
        if not self.metrics_computed:
            self.compute_metrics()
        
        # print('\n\nAccuracy : {:.2f}%'.format(self.global_simple_accuracy*100))
        # print('\n\nError : {:.2f}%'.format((1-self.global_simple_accuracy)*100))
        
        # print('\n\n## Detection ##')
        # print('\nPrecision : {:.2f}%'.format(self.global_detection_precision*100))
        # print('Recall : {:.2f}%'.format(self.global_detection_recall*100))
        
        # print('\n## Identification ##')
        # for i in range(9 if self.void_let_serve else 10):
        #     print('\n# {:s} #'.format(self.labels_map[i]))
        #     print('Precision : {:.2f}%'.format(self.global_identification_precision[i]*100))
        #     print('Recall : {:.2f}%'.format(self.global_identification_recall[i]*100))
        
        # print('\n## Mismatch ##')
        # print('\nPlayer 1 instead of 2: {:.2f}%'.format(self.player_mismatch[0]/self.player_mismatch[3]*100))
        # print('\nPlayer 2 instead of 1: {:.2f}%'.format(self.player_mismatch[1]/self.player_mismatch[2]*100))
        # print('\nBackhand instead of forehand: {:.2f}%'.format(self.hand_mismatch[0]/self.hand_mismatch[3]*100))
        # print('\nForehand instead of backhand: {:.2f}%'.format(self.hand_mismatch[1]/self.hand_mismatch[2]*100))
        
        labels_map = list(self.labels_map.values())

        def print_separator(width=65):
            print('-' * width)
        
        print('\n')
        print_separator()
        print(f"{'General Statistics':^65}")
        print_separator()
        print(f"{'Metric':<35}{'Value':>30}")
        print(f"{'Accuracy':<35}{self.global_simple_accuracy*100:>29.2f}%")
        print(f"{'Error':<35}{(1-self.global_simple_accuracy)*100:>29.2f}%")

        print_separator()
        print(f"{'Detection Statistics':^65}")
        print_separator()
        print(f"{'Metric':<35}{'Value':>30}")
        print(f"{'Precision':<35}{self.global_detection_precision*100:>29.2f}%")
        print(f"{'Recall':<35}{self.global_detection_recall*100:>29.2f}%")
        
        print_separator()
        print(f"{'Identification Statistics':^65}")
        print_separator()
        print(f"{'Label':<25}{'Precision':>20}{'Recall':>20}")
        for i in range(len(labels_map)):
            print(f"{labels_map[i]:<25}{self.global_identification_precision[i]*100:>19.2f}%{self.global_identification_recall[i]*100:>19.2f}%")
        
        print_separator()
        print(f"{'Mismatch Statistics':^65}")
        print_separator()
        print(f"{'Mismatch Type':<35}{'Percentage':>30}")
        print(f"{'Player 1 instead of 2':<35}{self.player_mismatch[0]/self.player_mismatch[3]*100:>29.2f}%")
        print(f"{'Player 2 instead of 1':<35}{self.player_mismatch[1]/self.player_mismatch[2]*100:>29.2f}%")
        print(f"{'Backhand instead of forehand':<35}{self.hand_mismatch[0]/self.hand_mismatch[3]*100:>29.2f}%")
        print(f"{'Forehand instead of backhand':<35}{self.hand_mismatch[1]/self.hand_mismatch[2]*100:>29.2f}%")
        print('\n')

    def get_metrics(self):
        
        if not self.metrics_computed:
            self.compute_metrics()
        
        return self.global_simple_accuracy, self.by_frame_simple_accuracy, self.global_detection_precision, self.global_detection_recall, self.by_frame_detection_precision, self.by_frame_detection_recall, self.global_identification_precision, self.global_identification_recall, self.by_frame_identification_precision, self.by_frame_identification_recall
        


        