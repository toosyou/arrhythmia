import numpy as np
from scipy.signal import find_peaks
import tensorflow as tf

def peak_confusion_matrix(y_true, y_pred, min_peak_distance, min_peak_height, max_delta):
    true_positive = np.zeros((y_true.shape[-1], ), dtype=int)
    false_negative = np.zeros((y_true.shape[-1], ), dtype=int)
    false_positive = np.zeros((y_true.shape[-1], ), dtype=int)

    for subject_true, subject_pred in zip(y_true, y_pred):
        pred_peaks, _ = find_peaks(subject_pred.max(axis=-1), distance=min_peak_distance, height=min_peak_height)
        pred_labels = np.argmax(subject_pred[pred_peaks], axis=-1)

        true_peaks, _ = find_peaks(subject_true.max(axis=-1), height=0.9)
        true_labels = np.argmax(subject_true[true_peaks], axis=-1)

        peak_matching = (np.abs(true_peaks[..., np.newaxis] - pred_peaks[np.newaxis, ...]) <= max_delta)
        label_matching = (true_labels[..., np.newaxis] == pred_labels[np.newaxis, ...])

        matched_gt_peak = np.logical_and(peak_matching, label_matching).any(axis=-1)

        np.add.at(true_positive, true_labels[matched_gt_peak], 1) # increase true_positive by one with index true_labels[matched_gt_peak]

        # remaining true peaks
        np.add.at(false_negative, true_labels[~matched_gt_peak], 1)

        # remaining predicted peak
        np.add.at(false_positive, pred_labels, 1)
        np.add.at(false_positive, true_labels[matched_gt_peak], -1)

    return true_positive, false_negative, false_positive

def peak_report(true_positive, false_negative, false_positive):
    recall = true_positive / (true_positive + false_negative)
    precision = true_positive / (true_positive + false_positive)
    f1_score = 2.0 * (precision * recall) / (precision + recall)
    return recall, precision, f1_score

def gpu_allow_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except:
            print('GPU allow growth failed!')
    return

if __name__ == "__main__":
    from train import MITLoader, DataGenerator

    mit_loader = MITLoader()
    training_set = DataGenerator(mit_loader, 'train', 64)
    X, y = training_set[0]
    peak_f1(100, 0.5, 10)(y, y)