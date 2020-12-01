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
from hg_blocks import create_hourglass_network, bottleneck_block

def focal_loss(length_head_ignore=0, length_tail_ignore=0, alpha=2, beta=4):
    def fl(y_true, y_pred):
        # ingore head and tail
        y_true = y_true[:, length_head_ignore: -length_tail_ignore, :]
        y_pred = y_pred[:, length_head_ignore: -length_tail_ignore, :]

        # clip values to prevent overflow
        y_pred = K.clip(y_pred, 1e-4, 1 - 1e-4)

        gt_mask = (y_true == 1)
        gt_loss = K.pow(1 - y_pred[gt_mask], alpha) * K.log(y_pred[gt_mask])
        focal_loss = K.pow(1 - y_true[~gt_mask], beta) * K.pow(y_pred[~gt_mask], alpha) * K.log(1 - y_pred[~gt_mask])
        return -(K.sum(gt_loss) + K.sum(focal_loss)) / K.sum(K.cast(gt_mask, 'float32')) / K.cast(K.shape(y_true)[0], 'float32')
    return fl

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('./config.ini')

    wandb.init('arrhythmia', entity=config['General']['entity'], 
                config={# data
                        'sampling_rate': 250,
                        'length_s': 12,
                        'head_ignore_s': 0.5,
                        'tail_ignore_s': 0.5,
                        'heatmap_std': 10,

                        'labels': [s.strip() for s in config['General']['labels'].split(',')],
                        'train_index' : [int(s.strip()) for s in config['MIT']['train_index'].split(',')],
                        'test_index' : [int(s.strip()) for s in config['MIT']['test_index'].split(',')],
                        'val_index' : [int(s.strip()) for s in config['MIT']['val_index'].split(',')],
                    
                        # model
                        'downsample_ratio': 1/4, # hourglass net
                        'number_hourglass_modules': 3,
                        'number_inner_channels': 16,
                        'hourglass_module_layers': 4,
                    
                        # training
                        'batch_size': 64,
    })
    wandb.config.update({
        # evaluate
        'min_peak_distance': int(100 * wandb.config.downsample_ratio),
        'peak_max_delta': int(100 / 1000 * 
                                wandb.config.sampling_rate * 
                                wandb.config.downsample_ratio), # 100ms
        'min_peak_height': 0.3,
    })

    mit_loader_train = MITLoader('train', wandb.config)
    mit_loader_val = MITLoader('valid', wandb.config)
    training_set = DataGenerator(mit_loader_train, wandb.config.batch_size)
    validation_set = DataGenerator(mit_loader_val, wandb.config.batch_size)

    print('#batch_train, #batch_valid: {}, {}'.format(len(training_set), len(validation_set)))

    model = create_hourglass_network(len(wandb.config.labels),
                                        wandb.config.number_hourglass_modules, 
                                        wandb.config.number_inner_channels,
                                        (int(wandb.config.sampling_rate * wandb.config.length_s), 1), 
                                    bottleneck_block,
                                    wandb.config.hourglass_module_layers)
    model.summary()
    model.compile(optimizer='adam', 
                    loss=focal_loss(length_head_ignore=int(wandb.config.sampling_rate * 
                                                            wandb.config.head_ignore_s *
                                                            wandb.config.downsample_ratio),
                                    length_tail_ignore=int(wandb.config.sampling_rate * 
                                                            wandb.config.tail_ignore_s*
                                                            wandb.config.downsample_ratio)),
                    run_eagerly=False)
                    
    model.fit(training_set, validation_data=validation_set, 
                epochs=500,
                callbacks=[
                    PerformanceLoger(training_set, validation_set,
                                        mit_loader_train.target_labels,
                                        wandb.config.min_peak_distance,
                                        wandb.config.min_peak_height,
                                        wandb.config.peak_max_delta),
                    LogBest(records=['val_loss', 'loss',] + 
                                        ['{}_f1_score'.format(l) for l in mit_loader_train.target_labels] + 
                                        ['val_{}_f1_score'.format(l) for l in mit_loader_train.target_labels] ),
                    WandbCallback(),
                    tf.keras.callbacks.EarlyStopping(patience=20)
                ], )