import os
from typing import Generator
import numpy as np
import configparser
from sklearn.model_selection import train_test_split
from scipy.interpolate import CubicSpline


import tensorflow as tf

import better_exceptions; better_exceptions.hook()

class MITLoader():
    def __init__(self, which_set, wandb_config):
        self.wandb_config = wandb_config

        self.config = configparser.ConfigParser()
        self.config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini'))
        self.config = self.config['MIT']

        self.sampling_rate = int(self.config['sampling_rate'])
        assert self.sampling_rate == wandb_config.sampling_rate, 'sampling rate mismatched!'

        self.length_segment = int(self.sampling_rate * wandb_config.length_s)
        self.output_length = int(self.length_segment * wandb_config.downsample_ratio)

        self.head_ignore = int(self.sampling_rate * wandb_config.head_ignore_s)
        self.tail_ignore = int(self.sampling_rate * wandb_config.tail_ignore_s)

        self.target_labels = np.array(wandb_config.labels)
        self.minor_labels = self.get_minor_labels()

        self.which_set = which_set

        self.index = {
            'train': np.array(wandb_config.train_index),
            'valid': np.array(wandb_config.val_index)
        }[which_set] 

        self.X, self.peak, self.label = self.load_signal()
        self.pos, self.pos_p = self.get_split()

    def data_resample(self, X, peak, orignal_rate, new_rate, mode):

        new_X = []
        new_peak = []
        for i in range(X.shape[0]):

            if mode == "Fourier":
                new_X.append(signal.resample(X[i], X[i].shape[0]*new_rate//orignal_rate))
                new_peak.append(peak[i]*new_rate//orignal_rate)

            elif mode == "interpolate":
                
                n = X[i].shape[0]
                T = n/orignal_rate
                m = int(new_rate * T +1)
                t = [(2*(i+1)-1)*T/(2*n) for i in range(n)]
                cs  = CubicSpline(t,X[i])
                t_p = [(2*(i+1)-1)*T/(2*m) for i in range(m)]
                new_X.append(cs(t_p))
                new_peak.append(int(peak[i]/orignal_rate*new_rate))

        new_X = np.array(new_X)
        new_peak = np.array(new_peak)

        return new_X, new_peak

    def load_signal(self):
        prefix = self.config['data_dir']

        X = np.load(os.path.join(prefix, 'MIT_data_denoise.npy'), allow_pickle=True)[self.index]
        peak = np.load(os.path.join(prefix, 'MIT_peak.npy'), allow_pickle=True)[self.index]
        label = np.load(os.path.join(prefix, 'MIT_label.npy'), allow_pickle=True)[self.index]

        X, peak = self.data_resample(X, peak, 360, 250, "interpolate")

        peak = [np.array(p) for p in peak]
        label = [np.array(l) for l in label]

        return X, peak, label

    def get_minor_labels(self):
        minor_labels = dict()
        for label in self.target_labels:
            minor_labels[label] = self.config['{}_labels'.format(label)].split(',')
            minor_labels[label] = np.array([t.strip() for t in minor_labels[label]])

        return minor_labels

    @staticmethod
    def gaussian(x, mean, std, normalize=False):
        unnormalized_gaussian = np.exp(-0.5 * ((x - mean)**2) / (std**2))
        if normalize:
            return (1. / (std * 2 * np.pi)) * unnormalized_gaussian
        return unnormalized_gaussian

    def gt_heatmap(self, peak, label):
        heatmap = np.zeros((self.output_length, self.target_labels.shape[0]))

        n_ = np.arange(self.output_length, dtype=float)
        for index_target, target_label in enumerate(self.target_labels):
            for minor_label in self.minor_labels[target_label]:
                for p in peak[label == minor_label] * self.wandb_config.downsample_ratio:
                    heatmap[:, index_target] += self.gaussian(n_, p, self.wandb_config.heatmap_std * self.wandb_config.downsample_ratio)

        return heatmap

    def get_usable_pos(self, subject_peak, subject_label):
        def usable_mask(label):
            mask = np.zeros_like(label, dtype=bool) # all False
            for tl in self.target_labels[:-1]:
                for ml in self.minor_labels[tl]:
                    mask = np.logical_or(mask, subject_label == ml)
            return mask

        usable_label_indices = np.where(usable_mask(subject_label))[0]
        pos = subject_peak[usable_label_indices]
        label = subject_label[usable_label_indices]

        # prevent pos from begin too close to the beginning and the end
        label   = label[pos > self.length_segment//2]
        pos     = pos[pos > self.length_segment//2]

        label   = label[pos < self.X.shape[1] - self.length_segment//2]
        pos     = pos[pos < self.X.shape[1] - self.length_segment//2]
        return pos, label

    def get_split(self, valid_ratio=0.05, random_state=42):
        poses = list()
        pos_p  = list()

        for index_subject in range(self.X.shape[0]):
            pos, label = self.get_usable_pos(self.peak[index_subject], self.label[index_subject])

            poses.append(pos)

            if self.which_set == "train":
                # calculate chosen probability for oversampling
                unique, counts = np.unique(label, return_counts=True)
                p = np.zeros((pos.shape[0], ), dtype=float)

                for uq, cnt in zip(unique, counts):
                    p[label == uq] = 1. / cnt / unique.shape[0]
                pos_p.append(p)
            else:
                # don't oversample validation set
                pos_p.append(np.ones((pos.shape[0], )) / pos.shape[0])
                
        return poses,pos_p

    def get_batch(self, pos, pos_p, batch_size):
        # random choose subject
        X, y = np.zeros((batch_size, self.length_segment, 1)), np.zeros((batch_size, self.output_length, self.target_labels.shape[0]))

        for batch_index, subject_index in enumerate(np.random.randint(len(pos), size=batch_size)):
            subject_pos = np.random.choice(pos[subject_index], p=pos_p[subject_index])
            subject_peak = self.peak[subject_index]
            subject_label = self.label[subject_index]

            start = subject_pos-self.length_segment//2
            end = start+self.length_segment

            peak_mask = np.logical_and(subject_peak >= start, subject_peak < end)

            X[batch_index] = self.X[subject_index, start:end]
            y[batch_index] = self.gt_heatmap(subject_peak[peak_mask] - start,
                                            subject_label[peak_mask])

        return X, y

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_loader, batch_size):
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.pos = self.data_loader.pos
        self.pos_p = self.data_loader.pos_p

        self.num_pos = sum([p.shape[0] for p in self.pos])
        self.shape_X = [self.batch_size, self.data_loader.length_segment, 1]
        self.shape_y = [self.batch_size, self.data_loader.length_segment, self.data_loader.target_labels.shape[0]]

    def __len__(self):
        return self.num_pos // self.batch_size

    def instance_norm(self, X):
        return (X - X.mean(axis=0)) / X.std(axis=0)
    
    def __getitem__(self, index):
        X, y = self.data_loader.get_batch(self.pos, self.pos_p, self.batch_size)
        return self.instance_norm(X), y


if __name__ == "__main__":
    mit_loader = MITLoader(which_set = 'train')
    train_data_generator = DataGenerator(mit_loader, 64)
    train_data_generator[0]