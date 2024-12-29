# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 14:06:18 2024

@author: anuvo
"""
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import warnings
warnings.filterwarnings("ignore")

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

void_let_serve = True

batch_size = 256
sequence_len = 60

torch.cuda.empty_cache()

# dataset = PingDataset(cvat_xml_dir='dataset', sequence_len=30, data_samples_dir='dataset/data_samples', void_let_serve=void_let_serve)
dataset = PingDataset(npy_dir='dataset/data_samples', sequence_len=sequence_len, void_let_serve=void_let_serve)

train_dataset, validation_dataset = dataset.train_validation_dataset(n_samples=-1, shuffle=True)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

train_stroke_count, train_nb_frames, validation_stroke_count, validation_nb_frames =  dataset.get_split_dataset_info(train_dataset, validation_dataset)

# model = Model_1(sequence_len=sequence_len, return_as_one=True).to(DEVICE)
# x_test = torch.rand(sequence_len, 93).to(DEVICE)
# print(model(x_test))

model = torch.load('models/model=1_d=128_h=4_n=4_bs=256_ns=-1_sl=60_shuffle_loss=stroke_detec+weighted_each_epoch=20+20+20+20+200_ams_grad.pt')

# model = Model_1(sequence_len=sequence_len, d_model=128, n_head=4, num_layers=4,
#                 stroke_logits=False, return_as_one=False, void_let_serve=void_let_serve).to(DEVICE)
# # x = torch.rand(batch_size, sequence_len, 93).to(DEVICE)
# # y_pred = model(x)
# # y_target = adapt_1_no_void(torch.rand(batch_size, sequence_len, 30).to(DEVICE))
# # print(adapt_output_1(*y_pred, void_let_serve).shape)
# # print([y_pred[i].shape for i in range(len(y_pred))])
# # print([y_target[i].shape for i in range(len(y_target))])

# # # model = Model_2(sequence_len=sequence_len, return_as_one=True).to(DEVICE)
# # # x_test = torch.rand(sequence_len, 93).to(DEVICE)
# # # print(model(x_test))

# # model = Model_2(sequence_len=sequence_len, d_model=128, n_head=8, num_layers=4, return_as_one=False, void_let_serve=void_let_serve).to(DEVICE)
# # x = torch.rand(batch_size, sequence_len, 93).to(DEVICE)
# # y_pred = model(x)
# # y_target = adapt_2_no_void(torch.rand(batch_size, sequence_len, 30).to(DEVICE))
# # print(adapt_output_2(*y_pred, void_let_serve).shape)
# # print([y_pred[i].shape for i in range(len(y_pred))])
# # print([y_target[i].shape for i in range(len(y_target))])


# # print(summary(model, input_size=(batch_size, sequence_len, 93)))

# ### TRAINING

# optimizer = torch.optim.Adam(model.parameters(), amsgrad=True)
# # # # optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# pos_weight = (train_nb_frames-train_stroke_count)/train_stroke_count if train_stroke_count !=0 else 1

# loss_function = Loss_1(void_let_serve=void_let_serve, stroke_identification=False, w_1=pos_weight)

# # loss_function = Loss_1_with_logits(void_let_serve=void_let_serve, stroke_identification=False, pos_weight=pos_weight)


# # # # loss_function = Loss_2(void_let_serve=void_let_serve)

# # # # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# train_loss_list, validation_loss_list, lr_list, best_model = train(model=model,
#                                                                    loss_function=loss_function,
#                                                                    optimizer=optimizer,
#                                                                    n_epochs=30,
#                                                                    training_dataloader=train_dataloader, validation_dataloader=validation_dataloader,
#                                                                    scheduler=None,
#                                                                    save_model=True, model_path='models/model=1_d=128_h=4_n=4_bs=256_ns=-1_sl=60_shuffle_loss=stroke_detec+weighted_each_epoch=20+20+20+20+200+30_ams_grad.pt',
#                                                                    save_fig=True, fig_path='fig_temp/model=1_d=128_h=4_n=4_bs=256_ns=-1_sl=60_shuffle_loss=stroke_detec+weighted_each_epoch=20+20+20+20+200+30_ams_grad')


accuracy = Accuracy(sequence_len=sequence_len, void_let_serve=void_let_serve)
evaluate_accuracy(model, validation_dataloader, accuracy)

global_detection_precision, global_detection_recall, by_frame_detection_precision, by_frame_detection_recall, global_identification_precision, global_identification_recall, by_frame_identification_precision, by_frame_identification_recall = accuracy.get_metrics()

accuracy.display_metrics()

# if best_model is not None:
#     accuracy = Accuracy(sequence_len=sequence_len, void_let_serve=void_let_serve)
#     evaluate_accuracy(best_model, validation_dataloader, accuracy)
    
#     best_stroke_detection, best_stroke_identification = accuracy.metrics()

# fig, ax1 = plt.subplots(figsize=(10, 6))
# ax1.plot(train_loss_list, label='Training Loss', color='tab:blue', marker='o')
# ax1.plot(validation_loss_list, label='Validation Loss', color='tab:orange', marker='o')
# ax1.set_xlabel('Epochs')
# ax1.set_ylabel('Loss', color='black')
# ax1.tick_params(axis='y', labelcolor='black')
# ax1.legend(loc='upper left')

# ax2 = ax1.twinx()
# ax2.plot(lr_list, label='Learning Rate', color='tab:green', linestyle='--', marker='x')
# ax2.set_ylabel('Learning Rate', color='tab:green')
# ax2.tick_params(axis='y', labelcolor='tab:green')
# ax2.legend(loc='upper right')
# ax2.set_yscale('log')

# plt.title('Training and Validation Loss with Learning Rate')
# ax1.grid(True, linestyle='--', alpha=0.7)

# plt.tight_layout()
# plt.show()

# train_loss, validation_loss = evaluate_loss(model, loss_function, train_dataloader, validation_dataloader)
