import os
from typing import Generator
import numpy as np
import configparser
from sklearn.model_selection import train_test_split

import tensorflow as tf

import better_exceptions; better_exceptions.hook()

class MITLoader():
    def __init__(self, wandb_config):
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

        self.X, self.peak, self.label = self.load_signal()
        self.train_pos, self.train_pos_p, self.valid_pos, self.valid_pos_p = self.get_split()

    def load_signal(self):
        prefix = self.config['data_dir']

        X = np.load(os.path.join(prefix, 'MIT_train_data_denoise.npy'), allow_pickle=True)
        peak = np.load(os.path.join(prefix, 'MIT_train_peak.npy'), allow_pickle=True)
        label = np.load(os.path.join(prefix, 'MIT_train_label.npy'), allow_pickle=True)

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
        train_poses, valid_poses = list(), list()
        train_pos_p, valid_pos_p = list(), list()

        for index_subject in range(self.X.shape[0]):
            up, label = self.get_usable_pos(self.peak[index_subject], self.label[index_subject])

            train_pos, valid_pos, train_labels, valid_labels = train_test_split(up, label, test_size=valid_ratio, random_state=random_state)

            # prevent train positions from being too close to valid positions
            for vp in valid_pos:
                keep_mask = np.logical_or(train_pos < vp-self.length_segment, train_pos > vp + self.length_segment)
                train_pos = train_pos[keep_mask]
                train_labels = train_labels[keep_mask]

            train_poses.append(train_pos)
            valid_poses.append(valid_pos)

            # calculate chosen probability for oversampling
            unique, counts = np.unique(train_labels, return_counts=True)
            p = np.zeros((train_pos.shape[0], ), dtype=float)

            for uq, cnt in zip(unique, counts):
                p[train_labels == uq] = 1. / cnt / unique.shape[0]
            train_pos_p.append(p)

            # don't oversample validation set
            valid_pos_p.append(np.ones((valid_pos.shape[0], )) / valid_pos.shape[0])

        return train_poses, train_pos_p, valid_poses, valid_pos_p

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
    def __init__(self, data_loader, which_set, batch_size):
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.pos = {
            'train': self.data_loader.train_pos,
            'valid': self.data_loader.valid_pos
        }[which_set]
        self.pos_p = {
            'train': self.data_loader.train_pos_p,
            'valid': self.data_loader.valid_pos_p
        }[which_set]

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
    mit_loader = MITLoader()
    train_data_generator = DataGenerator(mit_loader, 'train', 64)
    train_data_generator[0]