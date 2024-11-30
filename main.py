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
from loss import Loss_1, Loss_2
from accuracy import AccuracyTest

from train_test import train, evaluate

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu' #'cpu'

dataset = PingDataset(cvat_xml_dir='dataset_test', sequence_len=30, data_samples_dir='dataset_test/data_samples')

# dataset = PingDataset(npy_dir='dataset_test/data_samples', sequence_len=100)

# dataset.animate_sequence(index=4000, destination_dir='dataset_test/animation', save_mp4=True)
# dataset.animate_sample(file='ygfdz.npy', destination_dir='dataset_test/animation', save_mp4=True, nb_frames=100)

# batch_size = 64
# sequence_len = 30

# dataset = PingDataset(npy_dir='dataset_test/data_samples', sequence_len=sequence_len)

# train_dataset, test_dataset = dataset.train_test_dataset(n_samples=1024)

# train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


# # model = Model_1(sequence_len=sequence_len, return_as_one=True).to(DEVICE)
# # x_test = torch.rand(sequence_len, 93).to(DEVICE)
# # print(model(x_test))

# model = Model_1(sequence_len=sequence_len, return_as_one=False).to(DEVICE)
# # x_test = torch.rand(batch_size, sequence_len, 93).to(DEVICE)
# # print(adapt_output_1(*model(x_test)))
# # print(model(x_test)[-1])

# # model = Model_2(sequence_len=sequence_len, return_as_one=True).to(DEVICE)
# # x_test = torch.rand(sequence_len, 93).to(DEVICE)
# # print(model(x_test))

# # model = Model_2(sequence_len=sequence_len, return_as_one=False).to(DEVICE)
# # x_test = torch.rand(batch_size, sequence_len, 93).to(DEVICE)
# # print(adapt_output_2(*model(x_test)))

# # print(summary(model.to(DEVICE), input_size=(batch_size, sequence_len, 93)))

# optimizer = torch.optim.Adam(model.parameters())

# loss_function = Loss_1()
# loss_function = Loss_2()

# train(model=model, loss_function=loss_function, optimizer=optimizer, n_epochs=1,
        # train_dataloader=train_dataloader, test_dataloader=test_dataloader)

# torch.save(model, 'models/model_2_test.pt')


# model = torch.load('models/model_1_test.pt')

# accuracy = AccuracyTest(sequence_len=sequence_len, adapt=adapt_output_1)
# evaluate(model, test_dataloader, accuracy)

# event_detection, event_identification = accuracy.metrics()

# print(event_detection)
# print(event_identification[1])

