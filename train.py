import numpy as np
import configparser

import tensorflow as tf
from tensorflow.keras import backend as K
from train_utils import gpu_allow_growth; gpu_allow_growth()

import better_exceptions; better_exceptions.hook()

import wandb
from wandb.keras import WandbCallback

from data_utils import DataLoader, DataGenerator
from callbacks import PerformanceLoger, LogBest
from hg_blocks import create_hourglass_network, bottleneck_block

def focal_loss(length_head_ignore=0, length_tail_ignore=0, alpha=2, beta=4):
    def part_loss(true,pred):

        pred = K.clip(pred, 1e-4, 1 - 1e-4)

        gt_mask = (true == 1)
        gt_loss = K.pow(1 - pred[gt_mask], alpha) * K.log(pred[gt_mask]) 
        focal_loss = K.pow(1 - true[~gt_mask], beta) * K.pow(pred[~gt_mask], alpha) * K.log(1 - pred[~gt_mask])

        loss = -(K.sum(gt_loss) + K.sum(focal_loss)) / (K.sum(K.cast(gt_mask, 'float32')) + 1e-10) / K.cast(K.shape(true)[0], 'float32')
        nolabel_loss = -K.sum(focal_loss)/ K.cast(K.shape(true)[0], 'float32')

        num_pos = K.sum(K.cast(gt_mask, 'float32'))

        return loss *  K.cast((num_pos >  0),'float32')  + nolabel_loss * (1 - K.cast((num_pos >  0),'float32'))
 
    def fl(y_true, y_pred): 
        
        batch_size = K.shape(y_true)[0]

        N_mask = K.cast((y_true[batch_size//2:, length_head_ignore: -length_tail_ignore, 0] > 0),'float32')  

        true = y_true[batch_size//2:, length_head_ignore: -length_tail_ignore, 0]
        pred = y_pred[batch_size//2:, length_head_ignore: -length_tail_ignore, 0] + y_pred[batch_size//2:, length_head_ignore: -length_tail_ignore, 1] * N_mask
        NS_loss = part_loss(true,pred)

        true = y_true[batch_size//2:, length_head_ignore: -length_tail_ignore, 1]
        pred = y_pred[batch_size//2:, length_head_ignore: -length_tail_ignore, 1] * (1 - N_mask)
        S_loss = part_loss(true,pred)
 
        true = y_true[batch_size//2:, length_head_ignore: -length_tail_ignore, 2:]
        pred = y_pred[batch_size//2:, length_head_ignore: -length_tail_ignore, 2:] 
        other_loss = part_loss(true,pred)

        true = y_true[:batch_size//2, length_head_ignore: -length_tail_ignore, :]
        pred = y_pred[:batch_size//2, length_head_ignore: -length_tail_ignore, :]
        all_loss = part_loss(true,pred)

        return  NS_loss + S_loss + other_loss + all_loss 

    return fl


if __name__ == "__main__":  
    config = configparser.ConfigParser()
    config.read('./config.ini')

    wandb.init('arrhythmia-master', entity=config['General']['entity'], 
                config={# data
                        'sampling_rate': 250,
                        'length_s': 10,
                        'head_ignore_s': 0.5,
                        'tail_ignore_s': 0.5,
                        'heatmap_std': 10,
                        
                        'labels': [s.strip() for s in config['General']['labels'].split(',')],
                        'database' : [s.strip() for s in config['General']['database'].split(',')],
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

    mit_loader_train = DataLoader('MIT','train', wandb.config)
    mit_loader_val = DataLoader('MIT','valid', wandb.config)
    aha_loader_train = DataLoader('AHA','train', wandb.config)
    aha_loader_val = DataLoader('AHA','valid', wandb.config)

    training_set = DataGenerator([mit_loader_train,aha_loader_train], wandb.config.batch_size)
    validation_set = DataGenerator([mit_loader_val,aha_loader_val], wandb.config.batch_size)
    #mit_validation_set = DataGenerator(mit_loader_val, wandb.config.batch_size)
    #aha_training_set = DataGenerator(aha_loader_train, wandb.config.batch_size)
    #aha_validation_set = DataGenerator(aha_loader_val, wandb.config.batch_size)


    #training_set = DataGenerator(mit_loader_train, wandb.config.batch_size)
    #validation_set = DataGenerator(mit_loader_val, wandb.config.batch_size)


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