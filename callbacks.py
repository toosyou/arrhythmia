import numpy as np
from tqdm import tqdm
import tensorflow as tf
import wandb
import multiprocessing as mp

from train_utils import peak_confusion_matrix, peak_report

class PerformanceLoger(tf.keras.callbacks.Callback):
    def __init__(self, training_set, validation_set,
                    labels,
                    min_peak_distance, 
                    min_peak_height, 
                    max_delta, **kwargs):
        self.training_set = training_set
        self.validation_set = validation_set
        self.labels = labels
        self.min_peak_distance = min_peak_distance
        self.min_peak_height = min_peak_height
        self.max_delta = max_delta

        super().__init__(**kwargs)

    def log_evaluate(self, which_set, logs):
        predict_set = {
            'train': self.training_set,
            'valid': self.validation_set
        }[which_set]
        prefix = {
            'train': '',
            'valid': 'val_'
        }[which_set]
        
        #data_loader_list = predict_set.data_loader_list 
        X, y = predict_set.get_all_batch(predict_set.pos, predict_set.pos_p, 128)
        
        y_pred = self.model.predict(X, batch_size=128)
        y_pred = y_pred[-1] if isinstance(y_pred, list) else y_pred

        head_ignore, tail_ignore = predict_set.head_ignore, predict_set.tail_ignore
        true_positive, false_negative, false_positive = peak_confusion_matrix(y[:, head_ignore: -tail_ignore, :], 
                                                                                y_pred[:, head_ignore: -tail_ignore, :], 
                                                                                self.min_peak_distance,
                                                                                self.min_peak_height,
                                                                                self.max_delta)
        recall, precision, f1_score = peak_report(true_positive, false_negative, false_positive)

        for i, the_label in enumerate(self.labels):
            normalize_term = true_positive[i] + false_negative[i] + false_positive[i]
            logs['{}{}_TP'.format(prefix, the_label)] = true_positive[i] / normalize_term
            logs['{}{}_FN'.format(prefix, the_label)] = false_negative[i] / normalize_term
            logs['{}{}_FP'.format(prefix, the_label)] = false_positive[i] / normalize_term
            logs['{}{}_recall'.format(prefix, the_label)] = recall[i]
            logs['{}{}_precision'.format(prefix, the_label)] = precision[i]
            logs['{}{}_f1_score'.format(prefix, the_label)] = f1_score[i]

    def on_epoch_end(self, epochs, logs=None):
        self.log_evaluate('train', logs)
        self.log_evaluate('valid', logs)

class LogBest(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', records=['val_loss', 'loss', 'val_acc', 'acc']):
        self.monitor = monitor
        self.records = records

        setattr(self, 'best_' + self.monitor, np.inf if 'loss' in self.monitor else 0)
        super().__init__()

    def on_epoch_end(self, epoch, logs={}):
        if getattr(self, 'best_' + self.monitor) > logs.get(self.monitor): # update
            setattr(self, 'best_' + self.monitor, logs.get(self.monitor))

            log_dict = dict()
            for rs in self.records:
                log_dict['best_' + rs] = logs.get(rs)
            log_dict['best_epoch'] = epoch

            wandb.log(log_dict, commit=False)