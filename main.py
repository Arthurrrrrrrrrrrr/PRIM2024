# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 14:06:18 2024

@author: anuvo
"""
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchinfo import summary

from data_pipeline import PingDataset
from model import Model_1, Model_2, adapt_output_1, adapt_output_2
from loss import Loss_1, Loss_2, adapt_1_no_void, adapt_2_no_void
from accuracy import AccuracyTest

from train_validation import train, evaluate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

void_let_serve = True

batch_size = 64
sequence_len = 30

torch.cuda.empty_cache()

# dataset = PingDataset(cvat_xml_dir='dataset', sequence_len=30, data_samples_dir='dataset/data_samples', void_let_serve=void_let_serve)
dataset = PingDataset(npy_dir='dataset/data_samples', sequence_len=sequence_len, void_let_serve=void_let_serve)

train_dataset, validation_dataset = dataset.train_validation_dataset(n_samples=256)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)

# model = Model_1(sequence_len=sequence_len, return_as_one=True).to(DEVICE)
# x_test = torch.rand(sequence_len, 93).to(DEVICE)
# print(model(x_test))

# model = torch.load('models/model_1_0.02.pt')

# model = Model_1(sequence_len=sequence_len, d_model=128, n_head=8, num_layers=4, return_as_one=False, void_let_serve=void_let_serve).to(DEVICE)
# x = torch.rand(batch_size, sequence_len, 93).to(DEVICE)
# y_pred = model(x)
# y_target = adapt_1_no_void(torch.rand(batch_size, sequence_len, 30).to(DEVICE))
# print(adapt_output_1(*y_pred, void_let_serve).shape)
# print([y_pred[i].shape for i in range(len(y_pred))])
# print([y_target[i].shape for i in range(len(y_target))])

# # model = Model_2(sequence_len=sequence_len, return_as_one=True).to(DEVICE)
# # x_test = torch.rand(sequence_len, 93).to(DEVICE)
# # print(model(x_test))

model = Model_2(sequence_len=sequence_len, d_model=128, n_head=8, num_layers=4, return_as_one=False, void_let_serve=void_let_serve).to(DEVICE)
# x = torch.rand(batch_size, sequence_len, 93).to(DEVICE)
# y_pred = model(x)
# y_target = adapt_2_no_void(torch.rand(batch_size, sequence_len, 30).to(DEVICE))
# print(adapt_output_2(*y_pred, void_let_serve).shape)
# print([y_pred[i].shape for i in range(len(y_pred))])
# print([y_target[i].shape for i in range(len(y_target))])



# print(summary(model, input_size=(batch_size, sequence_len, 93)))

### TRAINING

optimizer = torch.optim.Adam(model.parameters())

loss_function = Loss_1(void_let_serve=void_let_serve)
# loss_function = Loss_2(void_let_serve=void_let_serve)

train_loss_list, validation_loss_list = train(model=model, loss_function=loss_function, optimizer=optimizer, n_epochs=2,
                                        train_dataloader=train_dataloader, validation_dataloader=validation_dataloader)

# # torch.save(model, 'models/model_1_0.02.pt')


accuracy = AccuracyTest(sequence_len=sequence_len, adapt=adapt_output_1, void_let_serve=void_let_serve)
evaluate(model, validation_dataloader, accuracy)

stroke_detection, stroke_identification = accuracy.metrics()

# print(stroke_detection)
# print(stroke_identification[1])

