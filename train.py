import numpy as np
import configparser

import tensorflow as tf
from tensorflow.keras import backend as K
from train_utils import gpu_allow_growth; gpu_allow_growth()

import better_exceptions; better_exceptions.hook()

import wandb
from wandb.keras import WandbCallback

from data_utils import MITLoader, DataGenerator
from callbacks import PerformanceLoger, LogBest
from model import unet
from hg_blocks import create_hourglass_network, bottleneck_block

def focal_loss(alpha=2, beta=4):
    def fl(y_true, y_pred):
        y_pred = K.clip(y_pred, 1e-4, 1 - 1e-4)
        gt_mask = (y_true == 1)
        gt_loss = K.pow(1 - y_pred[gt_mask], alpha) * K.log(y_pred[gt_mask])
        focal_loss = K.pow(1 - y_true[~gt_mask], beta) * K.pow(y_pred[~gt_mask], alpha) * K.log(1 - y_pred[~gt_mask])
        return -(K.sum(gt_loss) + K.sum(focal_loss)) / K.sum(K.cast(gt_mask, 'float32')) / K.cast(K.shape(y_true)[0], 'float32')
    return fl

if __name__ == "__main__":
    config = {
        'batch_size': 64,
        'min_peak_distance': int(100 * 0.25),
        'peak_max_delta': int(100 / 1000 * 360 * 0.25), # ~100ms
        'min_peak_height': 0.3,
    }
    wandb.init('arrhythmia', entity='toosyou', 
                config=config)

    mit_loader = MITLoader()
    training_set = DataGenerator(mit_loader, 'train', wandb.config.batch_size)
    validation_set = DataGenerator(mit_loader, 'valid', wandb.config.batch_size)

    print(len(training_set), len(validation_set))

    model = create_hourglass_network(mit_loader.target_labels.shape[0],
                                        2, 32, (mit_loader.length_segment, 1), bottleneck_block)
    model.summary()
    model.compile(optimizer='adam', loss=focal_loss(), run_eagerly=False)

    print(model.outputs[0].shape)

    model.fit(training_set, validation_data=validation_set, 
                epochs=500,
                callbacks=[
                    PerformanceLoger(training_set, validation_set,
                                        mit_loader.target_labels,
                                        wandb.config.min_peak_distance,
                                        wandb.config.min_peak_height,
                                        wandb.config.peak_max_delta),
                    LogBest(records=['val_loss', 'loss',] + 
                                        ['{}_f1_score'.format(l) for l in mit_loader.target_labels] + 
                                        ['val_{}_f1_score'.format(l) for l in mit_loader.target_labels] ),
                    WandbCallback(),
                    # tf.keras.callbacks.EarlyStopping(patience=10)
                ], )