import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

import better_exceptions; better_exceptions.hook()

import wandb
from wandb.keras import WandbCallback

from data_utils import MITLoader, DataGenerator
from train_utils import peak_f1, gpu_allow_growth; gpu_allow_growth()

from model import unet

def focal_loss(alpha=2, beta=4):
    def fl(y_true, y_pred):
        gt_mask = (y_true == 1)
        gt_loss = K.pow(1 - y_pred[gt_mask], alpha) * K.log(y_pred[gt_mask] + 1e-10)
        focal_loss = K.pow(1 - y_true[~gt_mask], beta) * K.pow(y_pred[~gt_mask], alpha) * K.log(1 - y_pred[~gt_mask] + 1e-10)
        return -(K.sum(gt_loss) + K.sum(focal_loss)) / K.sum(K.cast(gt_mask, 'float32')) / K.cast(K.shape(y_true)[0], 'float32')
    return fl

if __name__ == "__main__":
    config = {
        'batch_size': 64,
        'min_peak_distance': 100,
        'peak_max_delta': 10,
        'min_peak_height': 0.5,
    }
    wandb.init('arrhythmia', entity='toosyou', 
                config=config)

    mit_loader = MITLoader()
    training_set = DataGenerator(mit_loader, 'train', wandb.config.batch_size)
    validation_set = DataGenerator(mit_loader, 'valid', wandb.config.batch_size)

    print(len(training_set), len(validation_set))

    model = unet(training_set.shape_X[1:], training_set.shape_y[-1])
    model.summary()

    metrics=[peak_f1(wandb.config.min_peak_distance, 
                        wandb.config.min_peak_height,
                        wandb.config.peak_max_delta)]

    model.compile(optimizer='adam', loss=focal_loss(),
                    run_eagerly=True, metrics=metrics)

    model.fit(training_set, validation_data=validation_set, 
                epochs=500,
                callbacks=[
                    WandbCallback(),
                    # tf.keras.callbacks.EarlyStopping(patience=10)
                ], )