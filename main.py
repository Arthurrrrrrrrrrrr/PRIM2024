# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 14:06:18 2024

@author: anuvo
"""
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import warnings
warnings.filterwarnings("ignore")

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchinfo import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_pipeline import PingDataset
from model import Model_1, Model_2, adapt_output_1, adapt_output_2
from loss import Loss_1, Loss_1_with_logits, Loss_2, adapt_target_to_output_1_no_void, adapt_target_to_output_2_no_void
from accuracy import Accuracy
from train_validation import train, evaluate_accuracy, evaluate_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

void_let_serve = True
batch_size = 128
sequence_len = 60

n_samples = 4096

d_model = 128
n_head = 4
num_layers = 4
with_logits = True

n_epochs = 10

create_dataset = False
load_dataset = True
cvat_xml_dir = 'dataset'
data_samples_dir = 'dataset/data_samples'

load_model = False
load_model_dir = 'training/training_model=1_d=128_n=4_h=4_ns=-1_bs=128_task=detection_loss=weighted_+lrscheduler/'

load_training = False
load_training_dir = 'training/training_model=1_d=128_n=4_h=4_ns=-1_bs=128_task=identification_loss=logits+weighted/'

save_training = True
save_training_dir = 'training/training_model=1_d=128_n=4_h=4_ns=4096_bs=128_task=identification_loss=logits+weighted/'

weight_loss = True
stroke_identification = True

use_scheduler = False
clip_grad = False

get_data_info = True

compute_accuracy = True

concat_losses = True

plot_losses = False

if save_training:
    if not os.path.exists(save_training_dir): 
        os.makedirs(save_training_dir)

if load_training:
    data = torch.load(load_training_dir+'data')
    void_let_serve = data['void_let_serve']
    batch_size = data['batch_size']
    sequence_len = data['sequence_len']
    training_dataset = data['training_dataset']
    validation_dataset = data['validation_dataset']
    
    checkpoint = torch.load(load_training_dir+'checkpoint')
    epoch = checkpoint['epoch']
    old_training_loss_list = checkpoint['training_loss_list']
    old_validation_loss_list = checkpoint['validation_loss_list']
    old_lr_list = checkpoint['lr_list']
    model = checkpoint['model']
    best_model = checkpoint['best_model']
    optimizer = checkpoint['optimizer']
    if 'detection_loss_list' in checkpoint.keys() and 'identification_loss_list' in checkpoint.keys():
        old_detection_loss_list = checkpoint['detection_loss_list']
        old_identification_loss_list = checkpoint['identification_loss_list']
    else:
        old_detection_loss_list = []
        old_identification_loss_list = []
    
    # # Force value
    # batch_size = 1024
    
else:
    if load_model:
        checkpoint = torch.load(load_model_dir+'checkpoint')
        model = checkpoint['best_model']
    else:
        model = Model_1(sequence_len=sequence_len, d_model=d_model, n_head=n_head, num_layers=num_layers,
                        with_logits=with_logits, return_as_one=False, void_let_serve=void_let_serve).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), amsgrad=True)
    
if not concat_losses or not load_training:
    old_training_loss_list = []
    old_validation_loss_list = []
    old_lr_list = []
    old_detection_loss_list = []
    old_identification_loss_list = []
    
if create_dataset:
    dataset = PingDataset(cvat_xml_dir=cvat_xml_dir, sequence_len=sequence_len, data_samples_dir=data_samples_dir, void_let_serve=void_let_serve)
    
if load_dataset:
    dataset = PingDataset(npy_dir=data_samples_dir, sequence_len=sequence_len, void_let_serve=void_let_serve)

if create_dataset or load_dataset:
    training_dataset, validation_dataset = dataset.train_validation_dataset(n_samples=n_samples, shuffle=True)

if save_training:
    data = {'void_let_serve': void_let_serve,
            'batch_size': batch_size,
            'sequence_len': sequence_len,
            'training_dataset': training_dataset,
            'validation_dataset': validation_dataset}
    torch.save(data, save_training_dir+'data')
 
training_dataloader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=False)
validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)

if weight_loss or get_data_info:
    if not create_dataset and not load_dataset:
        dataset = PingDataset(npy_dir=data_samples_dir, sequence_len=sequence_len, void_let_serve=void_let_serve)
    train_stroke_count, train_nb_frames, validation_stroke_count, validation_nb_frames = dataset.get_split_dataset_info(training_dataset, validation_dataset)

if weight_loss:
    if with_logits:
        pos_weight = (train_nb_frames-train_stroke_count)/train_stroke_count if train_stroke_count !=0 else 1
        loss_function = Loss_1_with_logits(void_let_serve=void_let_serve, stroke_identification=stroke_identification, pos_weight=pos_weight)
    else:
        pos_weight = (train_nb_frames-train_stroke_count)/train_stroke_count if train_stroke_count !=0 else 1
        loss_function = Loss_1(void_let_serve=void_let_serve, stroke_identification=stroke_identification, w_1=pos_weight)
else:
    if with_logits:
        loss_function = Loss_1_with_logits(void_let_serve=void_let_serve, stroke_identification=stroke_identification)
    else:
        loss_function = Loss_1(void_let_serve=void_let_serve, stroke_identification=stroke_identification)

if use_scheduler :
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, threshold=1e-3)
else:
    scheduler = None

training_loss_list, validation_loss_list, lr_list, best_model, detection_loss_list, identification_loss_list = train(model=model, loss_function=loss_function, optimizer=optimizer,
                                                                                                                    n_epochs=n_epochs,
                                                                                                                    training_dataloader=training_dataloader,
                                                                                                                    validation_dataloader=validation_dataloader,
                                                                                                                    save_best_model=True, best_model_path=save_training_dir+'best_model',
                                                                                                                    save_fig=True, fig_path=save_training_dir+'fig',
                                                                                                                    scheduler=scheduler,
                                                                                                                    save_epoch=save_training, checkpoint_path=save_training_dir+'checkpoint',
                                                                                                                    training_loss_list=old_training_loss_list,
                                                                                                                    validation_loss_list=old_validation_loss_list,
                                                                                                                    learning_rate_list=old_lr_list,
                                                                                                                    clip_grad=clip_grad,
                                                                                                                    detection_loss_list=old_detection_loss_list, identification_loss_list=old_identification_loss_list)

if compute_accuracy:
    accuracy = Accuracy(sequence_len=sequence_len, void_let_serve=void_let_serve)
    evaluate_accuracy(model, validation_dataloader, accuracy)
    global_detection_precision, global_detection_recall, by_frame_detection_precision, by_frame_detection_recall, global_identification_precision, global_identification_recall, by_frame_identification_precision, by_frame_identification_recall = accuracy.get_metrics()
    accuracy.display_metrics()

    if best_model is not None:
        best_accuracy = Accuracy(sequence_len=sequence_len, void_let_serve=void_let_serve)
        evaluate_accuracy(best_model, validation_dataloader, best_accuracy)
        global_detection_precision, global_detection_recall, by_frame_detection_precision, by_frame_detection_recall, global_identification_precision, global_identification_recall, by_frame_identification_precision, by_frame_identification_recall = best_accuracy.get_metrics()
        best_accuracy.display_metrics()

if plot_losses:
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(training_loss_list, label='Training Loss', color='tab:blue', marker='o')
    ax1.plot(validation_loss_list, label='Validation Loss', color='tab:orange', marker='o')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()
    ax2.plot(lr_list, label='Learning Rate', color='tab:green', linestyle='--', marker='x')
    ax2.set_ylabel('Learning Rate', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.legend(loc='upper right')
    # ax2.set_yscale('log')
    
    plt.title('Training and Validation Loss with Learning Rate')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()