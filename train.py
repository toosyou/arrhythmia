import os
from typing import Generator
import numpy as np
import configparser
from sklearn.model_selection import train_test_split

import tensorflow as tf

import better_exceptions; better_exceptions.hook()

class MITLoader():
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini'))
        self.config = self.config['MIT']

        self.sampling_rate = int(self.config['sampling_rate'])
        self.heatmap_std = float(self.config['heatmap_std'])
        self.length_segment = int(float(self.config['length_s']) * 360)
        self.head_ignore = int(float(self.config['head_ignore_s']) * 360)
        self.tail_ignore = int(float(self.config['tail_ignore_s']) * 360)

        self.target_labels, self.minor_labels = self.read_target()

        self.X, self.peak, self.label = self.load_signal()
        self.train_pos, self.valid_pos = self.get_split()

    def load_signal(self):
        prefix = self.config['data_dir']

        X = np.load(os.path.join(prefix, 'MIT_train_data_denoise.npy'), allow_pickle=True)
        peak = np.load(os.path.join(prefix, 'MIT_train_peak.npy'), allow_pickle=True)
        label = np.load(os.path.join(prefix, 'MIT_train_label.npy'), allow_pickle=True)

        peak = [np.array(p) for p in peak]
        label = [np.array(l) for l in label]

        return X, peak, label

    def read_target(self):
        target_labels = self.config['labels'].split(',')
        target_labels = np.array([t.strip() for t in target_labels])

        minor_labels = dict()
        for label in target_labels:
            minor_labels[label] = self.config['{}_labels'.format(label)].split(',')
            minor_labels[label] = np.array([t.strip() for t in minor_labels[label]])

        return target_labels, minor_labels

    @staticmethod
    def gaussian(x, mean, std, normalize=False):
        unnormalized_gaussian = np.exp(-0.5 * ((x - mean)**2) / (std**2))
        if normalize:
            return (1. / (std * 2 * np.pi)) * unnormalized_gaussian
        return unnormalized_gaussian

    def gt_heatmap(self, peak, label):
        heatmap = np.zeros((self.length_segment, self.target_labels.shape[0]))

        n_ = np.arange(self.length_segment, dtype=float)
        for index_target, target_label in enumerate(self.target_labels):
            for minor_label in self.minor_labels[target_label]:
                for p in peak[label == minor_label]:
                    heatmap[:, index_target] += self.gaussian(n_, p, self.heatmap_std)

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

        # prevent pos from begin too close to the beginning and the end
        pos = pos[pos > self.length_segment//2]
        pos = pos[pos < self.X.shape[1] - self.length_segment//2]
        return pos

    def get_split(self, valid_ratio=0.1, random_state=42):
        train_poses, valid_poses = list(), list()

        for index_subject in range(self.X.shape[0]):
            up = self.get_usable_pos(self.peak[index_subject], self.label[index_subject])

            train_pos, valid_pos = train_test_split(up, test_size=valid_ratio, random_state=random_state)

            # prevent train positions from being too close to valid positions
            for vp in valid_pos:
                train_pos = train_pos[np.logical_or(train_pos < vp-self.length_segment//2, train_pos > vp + self.length_segment//2 )]

            train_poses.append(train_pos)
            valid_poses.append(valid_pos)

        return train_poses, valid_poses

    def get_batch(self, pos, batch_size):
        # random choose subject
        X, y = np.zeros((batch_size, self.length_segment, 1)), np.zeros((batch_size, self.length_segment, self.target_labels.shape[0]))

        for batch_index, subject_indice in enumerate(np.random.randint(len(pos), size=batch_size)):
            subject_pos = np.random.choice(pos[subject_indice])
            subject_peak = self.peak[subject_indice]
            subject_label = self.label[subject_indice]

            start = subject_pos-self.length_segment//2
            end = start+self.length_segment

            peak_mask = np.logical_and(subject_peak >= start, subject_peak < end)

            X[batch_index] = self.X[subject_indice, start:end]
            y[batch_index] = self.gt_heatmap(subject_peak[peak_mask] - start, 
                                            subject_label[peak_mask])

        return X, y

class DataGenerator():
    def __init__(self, data_loader, which_set, batch_size):
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.pos = {
            'train': self.data_loader.train_pos,
            'valid': self.data_loader.valid_pos
        }[which_set]

        self.num_pos = sum([p.shape[0] for p in self.pos])

    def __len__(self):
        return self.num_pos // self.batch_size

    def __getitem__(self, index):
        return self.data_loader.get_batch(self.pos, self.batch_size)

if __name__ == "__main__":
    mit_loader = MITLoader()

    train_data_generator = DataGenerator(mit_loader, 'train', 64)
    train_data_generator[0]
