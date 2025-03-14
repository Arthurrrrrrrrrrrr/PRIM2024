# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 13:26:59 2024

@author: anuvo
"""

import time
import torch
import warnings
import os
import random
import math
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import matplotlib.animation as animation
import matplotlib as mpl
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import train_test_split
from cvat_xml import BallTrack, TableTrack, PersonTrack, EventSequence

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

class PingDataset(Dataset):
    
    def __init__(self, sequence_len: int=None, output_len: int=None, output_offset: int=None, sequence_gap: int=None,
                 cvat_xml_dir: str=None, npy_dir: str=None, data_samples_dir: str=None, void_let_serve: bool=True):
        """
        Parameters
        ----------
        sequence_len : int
            Length of the sequences used as model inputs.
            The default is None.
        output_len : int
            Length of the output sequences.
            If None, is set equal to sequence_len.
            The default is None.
        output_offset : int
            Index (Python convention) of the first frame of the output sequence wrt the input sequence.
            If None, is set equal to 0.
            The default is None.
        sequence_gap : int
            Number of frames that separate two sequences.
            If None, is set equal to 1.
            The default is None.
        cvat_xml_dir : str, optional
            Path of the directory where the xml files using CVAT annotation are stored.
            Must contain subfolders 'inputs' and 'outputs'.
            The default is None.
        npy_dir : str, optional
            Path of the directory where the npy files are stored.
            The default is None.
        data_samples_dir : str
            Path of the directory where to store the samples.
            The default is None.
        void_let_serve: bool
            Assimilate void_serve to let_serve.
            The default is True.
        
        Raises
        ------
        ValueError
            No data or wrong data is provided.

        Returns
        -------
        None.

        """
        
        self.sequence_len = sequence_len
        self.output_len = output_len
        self.output_offset = output_offset
        self.sequence_gap = sequence_gap
        self.cvat_xml_dir = cvat_xml_dir
        self.npy_dir = npy_dir
        self.data_samples_dir = data_samples_dir
        self.void_let_serve = void_let_serve
        
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
            
            self.labels_map_inverted = dict(zip(self.labels_map.values(), self.labels_map.keys()))
            self.n_strokes = 9
            
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
            
            self.labels_map_inverted = dict(zip(self.labels_map.values(), self.labels_map.keys()))
            self.n_strokes = 10
            
        self.nb_sequences = None
        self.nb_sequences_per_sample = None
        
        if self.sequence_len is None:
            raise ValueError("'sequence_len' must be provided to cut the samples.")
        
        if self.output_offset is None:
            self.output_offset = 0
            
        if self.output_len is None:
            self.output_len = self.sequence_len
            
        if self.sequence_gap is None:
            self.sequence_gap = 1
                
        if self.output_offset < 0 or self.output_offset > self.sequence_len-1:
            raise ValueError("'output_offset' = {:d} must be between (inclusive) 0 and sequence_len - 1 = {:d}.").format(self.output_offset, self.sequence_len-1)
            
        if self.output_len < 1 or self.output_len > self.sequence_len:
            raise ValueError("'output_len' = {:d} must be between (inclusive) 1 and sequence_len = {:d}.").format(self.output_offset, self.sequence_len)
          
        if self.output_offset+self.output_len > self.sequence_len:
            raise ValueError("'output_offset' + 'output_len' = {:d} must be smaller (inclusive) than sequence_len = {:d}.").format(self.output_offset+self.output_len, self.sequence_len)
        
        if self.npy_dir is not None:
                                        
            samples = os.listdir(self.npy_dir)
            
            self.nb_sequences_per_sample = [0]
            void_let_serve = None
            
            for sample_idx, sample in enumerate(samples):
                
                sample_name = sample.split('.')[0]
                
                sample_array = np.load(os.path.join(self.npy_dir, sample))

                if void_let_serve is not None:
                    if void_let_serve != (sample_array.shape[1] == 102):
                        raise ValueError("Inconsistent number of strokes found between samples")
                else:
                    void_let_serve = (sample_array.shape[1] == 102)
                
                if sample_array.shape[0] < self.sequence_len:
                    raise ValueError("'sequence_len' value {:d} is less than {:d}, the size of sample {:s}."
                                     .format(self.sequence_len, sample_array.shape[0], sample_name))
                
                self.nb_sequences_per_sample.append((sample_array.shape[0] - self.sequence_len)//self.sequence_gap + 1)
                                
            self.nb_sequences = sum(self.nb_sequences_per_sample)
            self.void_let_serve = void_let_serve
            
            print("'void_let_serve' has been set to {:s} based on the samples.".format(str(void_let_serve)))
                
        elif self.cvat_xml_dir is not None:
            
            if self.sequence_len is None:
                raise ValueError("'sequence_len' must be provided to cut the samples.")
                
            elif self.data_samples_dir is None:
                raise ValueError("'data_samples_dir' must be provided to store the samples")
                
            else:
                
                self.npy_dir = os.path.join(self.data_samples_dir)
                if not os.path.exists(self.npy_dir): 
                    os.makedirs(self.npy_dir)
                
                inputs, outputs = os.listdir(os.path.join(self.cvat_xml_dir, 'inputs')), os.listdir(os.path.join(self.cvat_xml_dir, 'outputs'))
                nb_sample = len(inputs)
                
                self.nb_sequences_per_sample = [0]
                
                for sample_idx in range(nb_sample):

                    input_path = os.path.join(self.cvat_xml_dir,'inputs', inputs[sample_idx])
                    output_path = os.path.join(self.cvat_xml_dir, 'outputs', outputs[sample_idx])
                    
                    input_array, nb_frames = self.xml_input_to_array(input_path)
                    input_array = self.transform_inputs(input_array)
                    
                    #out of fov binary
                    for frame in range(nb_frames):
                        for i in range(31):
                            if True in np.isnan(input_array[frame, 3*i:3*i+2]):
                                input_array[frame, 3*i+2] = 1
                            else:
                                input_array[frame, 3*i+2] = 0
                    
                    for frame in range(nb_frames):
                        for i in range(93):    
                            if np.isnan(input_array[frame, i]):
                                input_array[frame, i]=0                   
                            
                    output_array = self.xml_output_to_array(output_path, nb_frames)
                    
                    sample_array = np.concatenate((input_array, output_array), axis=1)
                    
                    sample_name = inputs[sample_idx].split('.')[0]
                    
                    np.save(os.path.join(self.npy_dir, sample_name+'.npy'), sample_array)
                        
                    if sample_array.shape[0] < self.sequence_len:
                        raise ValueError("'sequence_len' value {:d} is less than {:d}, the size of sample {:s}."
                                         .format(self.sequence_len, sample_array.shape[0], sample_name))
                    
                    self.nb_sequences_per_sample.append((sample_array.shape[0] - self.sequence_len)//self.sequence_gap + 1)
                    
                self.nb_sequences = sum(self.nb_sequences_per_sample)

        else:
            raise ValueError("At least 'cvat_xml_dir' or 'npy_dir' must be provided.")
        
        # store the samples in GPU memory
        
        self.samples = []
        
        samples_names = os.listdir(self.npy_dir)
        for sample_name in samples_names:
            sample_array = np.load(os.path.join(self.npy_dir, sample_name))
            self.samples.append(torch.tensor(sample_array).type(torch.float).to(DEVICE))
            
        # # test full store
        # samples_names = os.listdir(self.npy_dir)
        # self.features, self.labels = [], []
        
        # for sample_idx, sample_name in enumerate(samples_names):
        #     sample_array = np.load(os.path.join(self.npy_dir, sample_name))
        #     sample_tensor = torch.tensor(sample_array).type(torch.float)
            
        #     for sequence_idx in range(self.nb_sequences_per_sample[sample_idx+1]):
        #         sequence = sample_tensor[sequence_idx*self.sequence_gap : sequence_idx*self.sequence_gap + self.sequence_len]
            
        #         self.features.append(sequence[:, :-self.n_strokes])
        #         self.labels.append(sequence[self.output_offset : self.output_offset+self.output_len, -self.n_strokes:])
            

    def __len__(self):
        
        return self.nb_sequences
    
    def __getitem__(self, index):
        
        sample_idx = np.argmax([1 if sum(self.nb_sequences_per_sample[:i+1])<=index<=sum(self.nb_sequences_per_sample[:i+2])-1 else 0 for i in range(len(self.nb_sequences_per_sample)-1)])
        
        sample_tensor = self.samples[sample_idx]

        sequence_idx = index-sum(self.nb_sequences_per_sample[:sample_idx+1])
        sequence = sample_tensor[sequence_idx*self.sequence_gap : sequence_idx*self.sequence_gap + self.sequence_len]
        
        feature, label = sequence[:, :-self.n_strokes], sequence[self.output_offset : self.output_offset+self.output_len, -self.n_strokes:]
        
        # feature = self.features[index]
        # label = self.labels[index]
        
        return feature, label
    
    def get_sample(self, sample: str):
        
        if '.npy' not in sample:
            sample += '.npy'
        
        if sample not in os.listdir(self.npy_dir):
            raise ValueError("'{:s}' does not exists.".format(sample))
            
        sample_array = np.load(os.path.join(self.npy_dir, sample))

        feature, label = sample_array[:, :-self.n_strokes], sample_array[:, -self.n_strokes:]
        
        return feature, label

    def xml_input_to_array(self, xml_path: str):
        
        xml_inputs = ET.parse(xml_path)
        
        player1 = [PersonTrack.load(track)
                   for track in xml_inputs.findall("track[@label='Person']/skeleton/attribute[@name='Role'][.='Player 1']/../..")]
        player2 = [PersonTrack.load(track)
                   for track in xml_inputs.findall("track[@label='Person']/skeleton/attribute[@name='Role'][.='Player 2']/../..")]
        ball = [BallTrack.load(track)
                 for track in xml_inputs.findall("track[@label='Ball']/skeleton/attribute[@name='Main'][.='true']/../..")]
        table = [TableTrack.load(track)
                 for track in xml_inputs.findall("track[@label='Table']")]
        
        objects_trackings = [player1, player2, ball, table]
        
        ### Get max frame from all objects
        max_frame = 0
        
        for object_trackings in objects_trackings:
            for object_tracking in object_trackings:
                frame = object_tracking.last_frame()
                if frame > max_frame:
                    max_frame = frame
        
        ### Add points in array of shape (4, total frames)
        nb_frames = max_frame + 1
        lengths_objects_labels = [13, 13, 1, 4]
        objects_points = np.zeros((3*sum(lengths_objects_labels), nb_frames))
        objects_points = np.vectorize(lambda x: np.nan)(objects_points) # points out of fov will value nan
        
        for object_idx, object_trackings in enumerate(objects_trackings):
            
            row_offset = 3*sum(lengths_objects_labels[:object_idx])
            
            for object_tracking in object_trackings:
                
                start_frame = object_tracking.last_frame()-len(object_tracking)+1
                
                for local_frame in range(len(object_tracking)):
                    
                    if object_tracking[local_frame] is not None:
                        
                        global_frame = start_frame + local_frame
                        
                        for point_label in range(lengths_objects_labels[object_idx]):
                            
                            objects_points[row_offset+3*point_label:row_offset+3*point_label+2, global_frame] = object_tracking[local_frame][point_label].numpy()
                 
        objects_points = np.transpose(objects_points)
        
        return objects_points, nb_frames
    
    def xml_output_to_array(self, xml_path: str, nb_frames: int):
    
        xml_outputs = ET.parse(xml_path)
        evt_sqc = EventSequence(xml_outputs)
        
        stroke_sequence = np.zeros((nb_frames, self.n_strokes))
    
        for frame in range(nb_frames):
            for stroke in evt_sqc[frame]:
                
                if self.void_let_serve and (stroke == 'void serve' or stroke == 'let serve'):
                    stroke = 'let serve or void serve'
                
                stroke_sequence[frame, self.labels_map_inverted[stroke]] = 1
        
        return stroke_sequence

    def sequence_cut(self, sample_array):
        
        nb_frames = sample_array.shape[0]
        sequences = []
        
        for frame in range(nb_frames - self.sequence_len + 1):
            sequences.append(sample_array[frame : frame + self.sequence_len])
            
        return sequences
    
    def animate_sequence(self, features, labels, index: int, destination_dir: str, nb_frames: int=None, frame_offset: int=None, save_mp4: bool=True, save_gif: bool=False):
        
        if nb_frames is None:
            nb_frames = self.sequence_len
            
            if frame_offset is not None:
                frame_offset = 0
                warnings.warn("\n frame_offset ignored because nb_frames has not been provided."
                              "\n frame_offset has been set to 0.")
                                          
            warnings.warn("\n nb_frames has been set to sequence length.")
            
        if frame_offset is None:
            frame_offset = 0
            warnings.warn("\n frame_offset has been set to 0.")
             
        elif nb_frames>self.sequence_len:
            raise ValueError("nb_frames of {:d} cannot be greater than sequence_len of {:d}."
                             .format(nb_frames, self.sequence_len))
            
        elif nb_frames+frame_offset>self.sequence_len:
            raise ValueError("nb_frames + frame_offset of {:d} cannot be greater than sequence_len of {:d}."
                             .format(nb_frames+frame_offset, self.sequence_len))
        
        # features, labels = self[index]
        sequence = np.concatenate((features.cpu().numpy(), labels.cpu().numpy()), axis=1)
        
        anim = self.animate(sequence, nb_frames, frame_offset)
                
        plt.rcParams['animation.ffmpeg_path'] ='FFmpeg/bin/ffmpeg.exe'
        
        if save_mp4:
            if not os.path.exists(destination_dir): 
                os.makedirs(destination_dir)
            
            FFwriter=animation.FFMpegWriter(fps=30)
            anim.save(os.path.join(destination_dir, 'sequence_'+str(index)+'_animated.mp4'), writer=FFwriter) 
        
        if save_gif:
            if not os.path.exists(destination_dir): 
                os.makedirs(destination_dir)
            
            FFwriter=animation.FFMpegWriter(fps=30)
            anim.save(os.path.join(destination_dir, 'sequence_'+str(index)+'_animated.gif'), writer=FFwriter)
        
        plt.close()
        
    def animate_sample(self, destination_dir: str, data: np.ndarray=None, infered_labels: np.ndarray=None, file: str=None, nb_frames: int=90, frame_offset: int=0, save_mp4: bool=True, save_gif: bool=False):
        
        if file is not None:
        
            sample_array = np.load(os.path.join(self.npy_dir, file))
        
            if nb_frames>sample_array.shape[0]:
                raise ValueError("nb_frames of {:d} cannot be greater than sample length of {:d}."
                                 .format(nb_frames, sample_array.shape[0]))
                
            elif nb_frames+frame_offset>sample_array.shape[0]:
                raise ValueError("nb_frames + frame_offset of {:d} cannot be greater than sample length of {:d}."
                                 .format(nb_frames+frame_offset, sample_array.shape[0]))
                
            anim = self.animate(sample_array, nb_frames, frame_offset)
            
            plt.rcParams['animation.ffmpeg_path'] ='FFmpeg\\bin\\ffmpeg.exe'
            
            if save_mp4:
                FFwriter=animation.FFMpegWriter(fps=30)
                anim.save(os.path.join(destination_dir, file.split('.')[0]+str(random.randint(100, 999))+'_animated.mp4'), writer=FFwriter) 
            
            if save_gif:
                FFwriter=animation.FFMpegWriter(fps=30)
                anim.save(os.path.join(destination_dir, file.split('.')[0]+str(random.randint(100, 999))+'_animated.gif'), writer=FFwriter)
            
            plt.close()
            
        elif data is not None:
            
            anim = self.animate(data=data, nb_frames=data.shape[0], frame_offset=frame_offset, infered_labels=infered_labels)
            
            plt.rcParams['animation.ffmpeg_path'] ='FFmpeg\\bin\\ffmpeg.exe'
            
            if save_mp4:
                FFwriter=animation.FFMpegWriter(fps=30)
                anim.save(os.path.join(destination_dir, 'data_'+str(random.randint(100, 999))+'_animated.mp4'), writer=FFwriter) 
            
            if save_gif:
                FFwriter=animation.FFMpegWriter(fps=30)
                anim.save(os.path.join(destination_dir, 'data_'+str(random.randint(100, 999))+'_animated.gif'), writer=FFwriter)
            
            plt.close()
            
        
    def animate(self, data: np.ndarray, nb_frames: int, frame_offset: int, infered_labels: np.ndarray=None):
        
        sample = data[:, 0:2]
        
        for i in range(1, 31):
            sample = np.concatenate((sample, data[:, 3*i:3*i+2]), axis=1)
            
        sample = np.concatenate((sample, data[:, -self.n_strokes:]), axis=1)
        plt.ioff()
        
        player1_color = 'blue'
        player2_color = 'red'
        ball_color = 'green'
        table_color = 'black'

        fig, ax = plt.subplots()
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        n_2D_lines = 4+2*14
        lines_colors = [player1_color for _ in range(14)] + [player2_color for _ in range(14)] + [table_color for _ in range(4)]
        
        scat = ax.scatter([], [], color=ball_color)
        lines = [ax.add_line(mpl.lines.Line2D([], [], color=lines_colors[i])) for i in range(n_2D_lines)]
        text = ax.text(0.99, 0.99, '',
                va="top",
                ha="right",
                transform=ax.transAxes)
        
        text_format = 'Target :\n{target}\nPrediction :\n{prediction}'
        
        self.target_temp = ''
        self.prediction_temp  = ''
        
        def init():
            scat.set_offsets(np.c_[[], []])
            for line in lines:
                line.set_data([], [])
            text.set_text('')

            return scat, *lines
        
        def update(frame):
            
            frame += frame_offset
            
            #player1
            #nose shoulders
            lines[0].set_data([sample[frame, 0], sample[frame, 2]], [sample[frame, 1], sample[frame, 3]])
            lines[1].set_data([sample[frame, 0], sample[frame, 4]], [sample[frame, 1], sample[frame, 5]])
            #shoulders hips
            lines[2].set_data([sample[frame, 2], sample[frame, 4]], [sample[frame, 3], sample[frame, 5]])
            lines[3].set_data([sample[frame, 2], sample[frame, 14]], [sample[frame, 3], sample[frame, 15]])
            lines[4].set_data([sample[frame, 4], sample[frame, 16]], [sample[frame, 5], sample[frame, 17]])
            lines[5].set_data([sample[frame, 14], sample[frame, 16]], [sample[frame, 15], sample[frame, 17]])
            #shoulders elbows
            lines[6].set_data([sample[frame, 2], sample[frame, 6]], [sample[frame, 3], sample[frame, 7]])
            lines[7].set_data([sample[frame, 4], sample[frame, 8]], [sample[frame, 5], sample[frame, 9]])
            #elbows wrists
            lines[8].set_data([sample[frame, 6], sample[frame, 10]], [sample[frame, 7], sample[frame, 11]])
            lines[9].set_data([sample[frame, 8], sample[frame, 12]], [sample[frame, 9], sample[frame, 13]])
            #hips knees
            lines[10].set_data([sample[frame, 14], sample[frame, 18]], [sample[frame, 15], sample[frame, 19]])
            lines[11].set_data([sample[frame, 16], sample[frame, 20]], [sample[frame, 17], sample[frame, 21]])
            #knees ankles
            lines[12].set_data([sample[frame, 18], sample[frame, 22]], [sample[frame, 19], sample[frame, 23]])
            lines[13].set_data([sample[frame, 20], sample[frame, 24]], [sample[frame, 21], sample[frame, 25]])
            
            #player2
            #nose shoulders
            lines[14].set_data([sample[frame, 26], sample[frame, 28]], [sample[frame, 27], sample[frame, 29]])
            lines[15].set_data([sample[frame, 26], sample[frame, 30]], [sample[frame, 27], sample[frame, 31]])
            #shoulders hips
            lines[16].set_data([sample[frame, 28], sample[frame, 30]], [sample[frame, 29], sample[frame, 31]])
            lines[17].set_data([sample[frame, 28], sample[frame, 40]], [sample[frame, 29], sample[frame, 41]])
            lines[18].set_data([sample[frame, 30], sample[frame, 42]], [sample[frame, 31], sample[frame, 43]])
            lines[19].set_data([sample[frame, 40], sample[frame, 42]], [sample[frame, 41], sample[frame, 43]])
            #shoulders elbows
            lines[20].set_data([sample[frame, 28], sample[frame, 32]], [sample[frame, 29], sample[frame, 33]])
            lines[21].set_data([sample[frame, 30], sample[frame, 34]], [sample[frame, 31], sample[frame, 35]])
            #elbows wrists
            lines[22].set_data([sample[frame, 32], sample[frame, 36]], [sample[frame, 33], sample[frame, 37]])
            lines[23].set_data([sample[frame, 34], sample[frame, 38]], [sample[frame, 35], sample[frame, 39]])
            #hips knees
            lines[24].set_data([sample[frame, 40], sample[frame, 44]], [sample[frame, 41], sample[frame, 45]])
            lines[25].set_data([sample[frame, 42], sample[frame, 46]], [sample[frame, 43], sample[frame, 47]])
            #knees ankles
            lines[26].set_data([sample[frame, 44], sample[frame, 48]], [sample[frame, 45], sample[frame, 49]])
            lines[27].set_data([sample[frame, 46], sample[frame, 50]], [sample[frame, 47], sample[frame, 51]])
            
            #ball
            scat.set_offsets(np.c_[sample[frame, 52], sample[frame, 53]])
            
            #table
            lines[28].set_data([sample[frame, 54], sample[frame, 56]], [sample[frame, 55], sample[frame, 57]])
            lines[29].set_data([sample[frame, 56], sample[frame, 58]], [sample[frame, 57], sample[frame, 59]])
            lines[30].set_data([sample[frame, 58], sample[frame, 60]], [sample[frame, 59], sample[frame, 61]])
            lines[31].set_data([sample[frame, 60], sample[frame, 54]], [sample[frame, 61], sample[frame, 55]])
            
            #strokes
            strokes = self.strokes_vector_to_text(sample[frame, -self.n_strokes:])
            if len(strokes)!=0:
                self.target_temp = '\n'.join(strokes)
                self.prediction_temp = ' '
                text.set_text(text_format.format(target=self.target_temp, prediction=self.prediction_temp))
            
            #infered strokes
            if infered_labels is not None:
                strokes = self.strokes_vector_to_text(infered_labels[frame, -self.n_strokes:])
                if len(strokes)!=0:
                    self.prediction_temp = '\n'.join(strokes)
                    text.set_text(text_format.format(target=self.target_temp, prediction=self.prediction_temp))
                        
            return scat, *lines
        
        anim = animation.FuncAnimation(fig, update, frames=nb_frames, init_func=init, blit=True)
        
        return anim
    
    def strokes_vector_to_text(self, vector):
        
        strokes = []
        for i in range(self.n_strokes):
            if vector[i]==1:
                strokes.append(self.labels_map[i])
                
        return strokes
    
    def train_validation_dataset(self, validation_size=0.25, n_samples=-1, shuffle=True):

        if n_samples + math.ceil(n_samples*validation_size) > len(self):
            raise ValueError("n_samples + math.ceil(n_samples*validation_size) cannot be greater than the dataset size")
            
        train_idx, validation_idx = train_test_split(list(range(len(self))), test_size=validation_size, shuffle=shuffle)
        train_dataset, validation_dataset = Subset(self, train_idx[:n_samples]), Subset(self, validation_idx[:math.ceil(n_samples*validation_size) if n_samples !=-1 else -1])
        
        return train_dataset, validation_dataset
    
    def get_dataset_info(self):

        stroke_count = 0
        player_1_count = 0
        player_2_count = 0
        serve_count = 0
        ball_pass_count = 0
        point_count = 0
        mistake_count = 0
        
        if self.void_let_serve:
            let_serve_void_serve_count = 0
        else:
            let_serve_count = 0
            void_serve_count = 0
            
        forehand_count = 0
        backhand_count = 0
            
        for _, y in DataLoader(self, batch_size=512, shuffle=False):

            stroke = (y.sum(axis=-1) > 0).int()
            
            player_1 = (y[:, :, 0] == 1).int().unsqueeze(-1)
            player_2 = (y[:, :, 1] == 1).int().unsqueeze(-1)
            serve = (y[:, :, 2] == 1).int().unsqueeze(-1)
            ball_pass = (y[:, :, 3] == 1).int().unsqueeze(-1)
            point = (y[:, :, 4] == 1).int().unsqueeze(-1)
            mistake = (y[:, :, 5] == 1).int().unsqueeze(-1)
            
            if self.void_let_serve:
                let_serve_void_serve = (y[:, :, 6] == 1).int().unsqueeze(-1)
                forehand = (y[:, :, 7] == 1).int().unsqueeze(-1)
                backhand = (y[:, :, 8] == 1).int().unsqueeze(-1)
            else:
                let_serve = (y[:, :, 6] == 1).int().unsqueeze(-1)
                void_serve = (y[:, :, 7] == 1).int().unsqueeze(-1)
                forehand = (y[:, :, 8] == 1).int().unsqueeze(-1)
                backhand = (y[:, :, 9] == 1).int().unsqueeze(-1)
            
            stroke_count += stroke.sum().item()
            player_1_count += player_1.sum().item()
            player_2_count += player_2.sum().item()
            serve_count += serve.sum().item()
            ball_pass_count += ball_pass.sum().item()
            point_count += point.sum().item()
            mistake_count += mistake.sum().item()
            
            if self.void_let_serve:
                let_serve_void_serve_count += let_serve_void_serve.sum().item()
            else:
                let_serve_count += let_serve.sum().item()
                void_serve_count += void_serve.sum().item()
                
            forehand_count += forehand.sum().item()
            backhand_count += backhand.sum().item()
            
        print("Strokes : {:d}/{:d}, {:.2f} %".format(stroke_count, (len(self)*self.sequence_len), 100*stroke_count/(len(self)*self.sequence_len)))
        print("Player 1 : {:d}/{:d}, {:.2f} %".format(player_1_count, (len(self)*self.sequence_len), 100*player_1_count/(len(self)*self.sequence_len)))
        print("Player 2 : {:d}/{:d}, {:.2f} %".format(player_2_count, (len(self)*self.sequence_len), 100*player_2_count/(len(self)*self.sequence_len)))
        print("Serve : {:d}/{:d}, {:.2f} %".format(serve_count, (len(self)*self.sequence_len), 100*serve_count/(len(self)*self.sequence_len)))
        print("Ball pass : {:d}/{:d}, {:.2f} %".format(ball_pass_count, (len(self)*self.sequence_len), 100*ball_pass_count/(len(self)*self.sequence_len)))
        print("Point : {:d}/{:d}, {:.2f} %".format(point_count, (len(self)*self.sequence_len), 100*point_count/(len(self)*self.sequence_len)))
        print("Mistake : {:d}/{:d}, {:.2f} %".format(mistake_count, (len(self)*self.sequence_len), 100*mistake_count/(len(self)*self.sequence_len)))
        
        if self.void_let_serve:
            print("Let serve or Void serve : {:d}/{:d}, {:.2f} %".format(let_serve_void_serve_count, (len(self)*self.sequence_len), 100*let_serve_void_serve_count/(len(self)*self.sequence_len)))
        else:
            print("Let serve : {:d}/{:d}, {:.2f} %".format(let_serve_count, (len(self)*self.sequence_len), 100*let_serve_count/(len(self)*self.sequence_len)))
            print("Void serve : {:d}/{:d}, {:.2f} %".format(void_serve_count, (len(self)*self.sequence_len), 100*void_serve_count/(len(self)*self.sequence_len)))
            
        print("Forehand : {:d}/{:d}, {:.2f} %".format(forehand_count, (len(self)*self.sequence_len), 100*forehand_count/(len(self)*self.sequence_len)))
        print("Backhand : {:d}/{:d}, {:.2f} %".format(backhand_count, (len(self)*self.sequence_len), 100*backhand_count/(len(self)*self.sequence_len)))
        
    def get_split_dataset_info(self, train_dataset, validation_dataset):

        stroke_count = 0
        player_1_count = 0
        player_2_count = 0
        serve_count = 0
        ball_pass_count = 0
        point_count = 0
        mistake_count = 0
        
        if self.void_let_serve:
            let_serve_void_serve_count = 0
        else:
            let_serve_count = 0
            void_serve_count = 0
            
        forehand_count = 0
        backhand_count = 0
            
        for _, y in DataLoader(train_dataset, batch_size=512, shuffle=False):

            stroke = (y.sum(axis=-1) > 0).int()
            
            player_1 = (y[:, :, 0] == 1).int().unsqueeze(-1)
            player_2 = (y[:, :, 1] == 1).int().unsqueeze(-1)
            serve = (y[:, :, 2] == 1).int().unsqueeze(-1)
            ball_pass = (y[:, :, 3] == 1).int().unsqueeze(-1)
            point = (y[:, :, 4] == 1).int().unsqueeze(-1)
            mistake = (y[:, :, 5] == 1).int().unsqueeze(-1)
            
            if self.void_let_serve:
                let_serve_void_serve = (y[:, :, 6] == 1).int().unsqueeze(-1)
                forehand = (y[:, :, 7] == 1).int().unsqueeze(-1)
                backhand = (y[:, :, 8] == 1).int().unsqueeze(-1)
            else:
                let_serve = (y[:, :, 6] == 1).int().unsqueeze(-1)
                void_serve = (y[:, :, 7] == 1).int().unsqueeze(-1)
                forehand = (y[:, :, 8] == 1).int().unsqueeze(-1)
                backhand = (y[:, :, 9] == 1).int().unsqueeze(-1)
            
            stroke_count += stroke.sum().item()
            player_1_count += player_1.sum().item()
            player_2_count += player_2.sum().item()
            serve_count += serve.sum().item()
            ball_pass_count += ball_pass.sum().item()
            point_count += point.sum().item()
            mistake_count += mistake.sum().item()
            
            if self.void_let_serve:
                let_serve_void_serve_count += let_serve_void_serve.sum().item()
            else:
                let_serve_count += let_serve.sum().item()
                void_serve_count += void_serve.sum().item()
                
            forehand_count += forehand.sum().item()
            backhand_count += backhand.sum().item()
            
        print("\n## Training ##")
            
        print("Strokes : {:d}/{:d}, {:.2f} %".format(stroke_count, (len(train_dataset)*self.sequence_len), 100*stroke_count/(len(train_dataset)*self.sequence_len)))
        print("Player 1 : {:d}/{:d}, {:.2f} %".format(player_1_count, (len(train_dataset)*self.sequence_len), 100*player_1_count/(len(train_dataset)*self.sequence_len)))
        print("Player 2 : {:d}/{:d}, {:.2f} %".format(player_2_count, (len(train_dataset)*self.sequence_len), 100*player_2_count/(len(train_dataset)*self.sequence_len)))
        print("Serve : {:d}/{:d}, {:.2f} %".format(serve_count, (len(train_dataset)*self.sequence_len), 100*serve_count/(len(train_dataset)*self.sequence_len)))
        print("Ball pass : {:d}/{:d}, {:.2f} %".format(ball_pass_count, (len(train_dataset)*self.sequence_len), 100*ball_pass_count/(len(train_dataset)*self.sequence_len)))
        print("Point : {:d}/{:d}, {:.2f} %".format(point_count, (len(train_dataset)*self.sequence_len), 100*point_count/(len(train_dataset)*self.sequence_len)))
        print("Mistake : {:d}/{:d}, {:.2f} %".format(mistake_count, (len(train_dataset)*self.sequence_len), 100*mistake_count/(len(train_dataset)*self.sequence_len)))
        
        if self.void_let_serve:
            print("Let serve or Void serve : {:d}/{:d}, {:.2f} %".format(let_serve_void_serve_count, (len(train_dataset)*self.sequence_len), 100*let_serve_void_serve_count/(len(train_dataset)*self.sequence_len)))
        else:
            print("Let serve : {:d}/{:d}, {:.2f} %".format(let_serve_count, (len(train_dataset)*self.sequence_len), 100*let_serve_count/(len(train_dataset)*self.sequence_len)))
            print("Void serve : {:d}/{:d}, {:.2f} %".format(void_serve_count, (len(train_dataset)*self.sequence_len), 100*void_serve_count/(len(train_dataset)*self.sequence_len)))
            
        print("Forehand : {:d}/{:d}, {:.2f} %".format(forehand_count, (len(train_dataset)*self.sequence_len), 100*forehand_count/(len(train_dataset)*self.sequence_len)))
        print("Backhand : {:d}/{:d}, {:.2f} %".format(backhand_count, (len(train_dataset)*self.sequence_len), 100*backhand_count/(len(train_dataset)*self.sequence_len)))
        
        print("\n## Validation ##")
        
        validation_stroke_count = 0
        player_1_count = 0
        player_2_count = 0
        serve_count = 0
        ball_pass_count = 0
        point_count = 0
        mistake_count = 0
        
        if self.void_let_serve:
            let_serve_void_serve_count = 0
        else:
            let_serve_count = 0
            void_serve_count = 0
            
        forehand_count = 0
        backhand_count = 0
            
        for _, y in DataLoader(validation_dataset, batch_size=512, shuffle=False):

            stroke = (y.sum(axis=-1) > 0).int()
            
            player_1 = (y[:, :, 0] == 1).int().unsqueeze(-1)
            player_2 = (y[:, :, 1] == 1).int().unsqueeze(-1)
            serve = (y[:, :, 2] == 1).int().unsqueeze(-1)
            ball_pass = (y[:, :, 3] == 1).int().unsqueeze(-1)
            point = (y[:, :, 4] == 1).int().unsqueeze(-1)
            mistake = (y[:, :, 5] == 1).int().unsqueeze(-1)
            
            if self.void_let_serve:
                let_serve_void_serve = (y[:, :, 6] == 1).int().unsqueeze(-1)
                forehand = (y[:, :, 7] == 1).int().unsqueeze(-1)
                backhand = (y[:, :, 8] == 1).int().unsqueeze(-1)
            else:
                let_serve = (y[:, :, 6] == 1).int().unsqueeze(-1)
                void_serve = (y[:, :, 7] == 1).int().unsqueeze(-1)
                forehand = (y[:, :, 8] == 1).int().unsqueeze(-1)
                backhand = (y[:, :, 9] == 1).int().unsqueeze(-1)
            
            validation_stroke_count += stroke.sum().item()
            player_1_count += player_1.sum().item()
            player_2_count += player_2.sum().item()
            serve_count += serve.sum().item()
            ball_pass_count += ball_pass.sum().item()
            point_count += point.sum().item()
            mistake_count += mistake.sum().item()
            
            if self.void_let_serve:
                let_serve_void_serve_count += let_serve_void_serve.sum().item()
            else:
                let_serve_count += let_serve.sum().item()
                void_serve_count += void_serve.sum().item()
                
            forehand_count += forehand.sum().item()
            backhand_count += backhand.sum().item()
            
        print("Strokes : {:d}/{:d}, {:.2f} %".format(validation_stroke_count, (len(validation_dataset)*self.sequence_len), 100*validation_stroke_count/(len(validation_dataset)*self.sequence_len)))
        print("Player 1 : {:d}/{:d}, {:.2f} %".format(player_1_count, (len(validation_dataset)*self.sequence_len), 100*player_1_count/(len(validation_dataset)*self.sequence_len)))
        print("Player 2 : {:d}/{:d}, {:.2f} %".format(player_2_count, (len(validation_dataset)*self.sequence_len), 100*player_2_count/(len(validation_dataset)*self.sequence_len)))
        print("Serve : {:d}/{:d}, {:.2f} %".format(serve_count, (len(validation_dataset)*self.sequence_len), 100*serve_count/(len(validation_dataset)*self.sequence_len)))
        print("Ball pass : {:d}/{:d}, {:.2f} %".format(ball_pass_count, (len(validation_dataset)*self.sequence_len), 100*ball_pass_count/(len(validation_dataset)*self.sequence_len)))
        print("Point : {:d}/{:d}, {:.2f} %".format(point_count, (len(validation_dataset)*self.sequence_len), 100*point_count/(len(validation_dataset)*self.sequence_len)))
        print("Mistake : {:d}/{:d}, {:.2f} %".format(mistake_count, (len(validation_dataset)*self.sequence_len), 100*mistake_count/(len(validation_dataset)*self.sequence_len)))
        
        if self.void_let_serve:
            print("Let serve or Void serve : {:d}/{:d}, {:.2f} %".format(let_serve_void_serve_count, (len(validation_dataset)*self.sequence_len), 100*let_serve_void_serve_count/(len(validation_dataset)*self.sequence_len)))
        else:
            print("Let serve : {:d}/{:d}, {:.2f} %".format(let_serve_count, (len(validation_dataset)*self.sequence_len), 100*let_serve_count/(len(validation_dataset)*self.sequence_len)))
            print("Void serve : {:d}/{:d}, {:.2f} %".format(void_serve_count, (len(validation_dataset)*self.sequence_len), 100*void_serve_count/(len(validation_dataset)*self.sequence_len)))
            
        print("Forehand : {:d}/{:d}, {:.2f} %".format(forehand_count, (len(validation_dataset)*self.sequence_len), 100*forehand_count/(len(validation_dataset)*self.sequence_len)))
        print("Backhand : {:d}/{:d}, {:.2f} %\n".format(backhand_count, (len(validation_dataset)*self.sequence_len), 100*backhand_count/(len(validation_dataset)*self.sequence_len)))
        
        return stroke_count, len(train_dataset)*self.sequence_len, validation_stroke_count, len(validation_dataset)*self.sequence_len
        
    @staticmethod
    def transform_inputs(objects_points, rotate=True, scale_min_max=True):
        
        ### 180° rotation about (0, 0)
        if rotate:
            objects_points[:, ::3] = -objects_points[:, ::3]
            objects_points[:, 1::3] = -objects_points[:, 1::3]
                    
        ### Scale min max
        if scale_min_max:
            min_x, min_y = np.nanmin(objects_points[:, ::3]), np.nanmin(objects_points[:, 1::3])
            max_x, max_y = np.nanmax(objects_points[:, ::3]), np.nanmax(objects_points[:, 1::3])
            objects_points[:, ::3] = (objects_points[:, ::3] - min_x)/(max_x - min_x)
            objects_points[:, 1::3] = (objects_points[:, 1::3] - min_y)/(max_y - min_y)
        
        return objects_points                
    
    


