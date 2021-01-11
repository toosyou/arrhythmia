import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import configparser

from test_utils.denoise import ekg_denoise,baseline_wander_removal
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
from sklearn.metrics import confusion_matrix
from test_utils.beatbybeat import beatbybeat
from scipy import signal


def xml_reader(file):
    
    xml = ET.parse(file)
    root = xml.getroot()
    obj = root.findall('Signals')
    data = obj[0][0][1].text
    data = data.split('|')
    X = []
    for d in data:
        if d != "負無窮大":
            X.append(float(d))


    return np.array(X)

def signal_denoise(X, fs):

    original_std = X.std()
    X *= (500. / original_std)
    if X.shape[0] % 2 == 0:
        X = ekg_denoise(X.reshape(1,X.shape[0]),fs,1)
    else:
        X = ekg_denoise(X[:-1].reshape(1,X[:-1].shape[0]),fs,1)
    X *= (original_std / 500.)

    X = (X - np.mean(X))/np.std(X)
    X = X.reshape(X.shape[1],1)

    return X

def data_resample(X, orignal_rate, new_rate, mode):
    if mode == "Fourier":
        return signal.resample(X, X.shape[0]//orignal_rate*new_rate)

    elif mode == "interpolate":
        n = X.shape[0]
        T = n/orignal_rate
        m = int(new_rate * T +0.5)
        t = [(2*(i+1)-1)*T/(2*n) for i in range(n)]
        cs  = CubicSpline(t,X)
        t_p = [(2*(i+1)-1)*T/(2*m) for i in range(m)]

        return np.array(cs(t_p))

def preprocessing(X_original, orignal_rate, true_peaks):

    X = data_resample(X_original, orignal_rate, 250, "interpolate")
    X = signal_denoise(X,250)
    
    if true_peaks is not None:
        true_peaks = np.array(true_peaks/orignal_rate*250, dtype="int")

    return X, true_peaks
    

def adjust_label(pred_peaks,pred_labels,ratio):
    for p in range(1,pred_peaks.shape[0]-1,1):
        if pred_labels[p] == 0 and pred_labels[p+1] ==0 and pred_labels[p-1] == 0 and (pred_peaks[p+1]- pred_peaks[p])/(pred_peaks[p]-pred_peaks[p-1]) > ratio:
            pred_labels[p] = 1
        if pred_labels[p] == 1 and (pred_peaks[p+1]-pred_peaks[p])/(pred_peaks[p]-pred_peaks[p-1]) < ratio:
            pred_labels[p] = 0
#         if pred_labels[p] == 2 and pred_labels[p+1] ==0 and pred_labels[p-1] == 0 and (pred_peaks[p+1]-pred_peaks[p])/(pred_peaks[p]-pred_peaks[p-1]) < ratio-0.1:
#             pred_labels[p] = 0
#         if pred_labels[p] == 3 and pred_labels[p+1] ==0 and pred_labels[p-1] == 0 and (pred_peaks[p+1]-pred_peaks[p])/(pred_peaks[p]-pred_peaks[p-1]) < ratio-0.1:
#             pred_labels[p] = 0
        
    
    return pred_labels

def visualization(X,pred_peaks,pred_labels,cl, true_peaks=None,true_labels = None,pic_num = 5):
    
    num  = len(X)
    if num > pic_num:
        num = pic_num
    for n in range(num):
        peaks = pred_peaks[n][1:-1]
        labels = pred_labels[n][1:-1]
        fig, ax = plt.subplots(figsize=(20, 2))
        ax.title.set_text('predict {}'.format(n))
        ax.plot(X[n])
        color = ['bo','co','ro','go','ko']
        for i in range(len(cl)):
            pos = np.where(labels == i)[0]
            ax.plot(peaks[pos], X[n][peaks[pos]], color[i], label=cl[i])
        leg = ax.legend()

        if (true_peaks is not None) and (true_labels is not None):
            fig, ax = plt.subplots(figsize=(20, 2))
            ax.title.set_text('gt {}'.format(n))
            ax.plot(X[n])
            gt_peaks = true_peaks[n][1:-1]
            gt_labels = true_labels[n][1:-1]
            for i in range(len(cl)):
                gt_pos = np.where(gt_labels == cl[i])[0]
                ax.plot(gt_peaks[gt_pos], X[n][gt_peaks[gt_pos]], color[i], label=cl[i])

            leg = ax.legend()

    return 

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    
    # Begin CHANGES
    fst_empty_cell = (columnwidth-3)//2 * " " + "t/p" + (columnwidth-3)//2 * " "
    
    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")
    # End CHANGES
    
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
        
    print()
    # Print rows
    for i, label1 in enumerate(labels[:-2]):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1d".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


def performance(pred_peaks,pred_labels,true_peaks,true_labels,cl):
    
    ## return confusion matrix
    pair, miss, extra = beatbybeat(250, pred_peaks, true_peaks)
    true_labels_ = []
    for l in true_labels:
        true_labels_.append(cl.index(l))

    true_labels_ = np.array(true_labels_)
    C = np.zeros((len(cl)+2,len(cl)+2))
    
    if pair.shape[0] > 0:
        C[:len(cl),:len(cl)] = confusion_matrix(true_labels_[pair[:,0]],pred_labels[pair[:,1]],labels= [i for i in range(len(cl))])

    for m in miss:
        C[cl.index(true_labels[m]),-2] += 1

    for e in extra:
        C[pred_labels[e],-1] += 1
    
    return C


def deal_minor_labels(config,target_labels,peaks,labels):

    def get_minor_labels(target_labels,config):
        minor_labels = dict()

        for label in target_labels:
            minor_labels[label] = config['{}_labels'.format(label)].split(',')
            minor_labels[label] = np.array([t.strip() for t in minor_labels[label]])

        return minor_labels
    
    minor_labels = get_minor_labels(target_labels,config)
    
    new_peaks ,new_labels = list(),list()
    
    for i in range(peaks.shape[0]):
        _new_peaks ,_new_labels = list(),list()
        for p,l in zip(peaks[i],labels[i]):
            for label in target_labels:
                if l in minor_labels[label]:
                    _new_labels.append(label)
                    _new_peaks.append(p)
                    
        new_labels.append(np.array(_new_labels))
        new_peaks.append(np.array(_new_peaks))
        
    return new_peaks,new_labels
    

def predict(X_original, orignal_rate, model, true_peaks = None, true_labels = None, cl = None, adjust = True, ratio = 1.2):
    
    X_original, true_peaks = preprocessing(X_original, orignal_rate, true_peaks)
    
    print(X_original.shape)

    length = np.array(X_original).shape[0] 
    predict_len = 2500
    num = length//predict_len
    
    X_signal = []
    True_peaks = []
    True_labels = []
    start = 0
    end = 500
    
    while True:
        
        start = end - 500
        end = start + predict_len
        
        if end > length:
            break
        
        X_signal.append(X_original[start:end])
        
        if (true_peaks is not None) and (true_labels is not None):
            s = np.where(true_peaks >= start)[0]
            if len(s) == 0:
                continue
            e = np.where(true_peaks < end)[0]
            True_peaks.append(true_peaks[s[0]:e[-1]+1] - start)
            True_labels.append(true_labels[s[0]:e[-1]+1])
            
    
    Predict_peaks = []
    Predict_labels =  []
    
    for X in X_signal:

        y_pred = model.predict(X[np.newaxis,...])
        y_pred = y_pred[-1] if isinstance(y_pred, list) else y_pred
        y_pred = np.array(y_pred)
        pred_peaks, _ = find_peaks(y_pred[0].max(axis=-1), distance=25, height=0.2)
        pred_labels = np.argmax(y_pred[0][pred_peaks], axis=-1)

        if adjust:
            pred_labels = adjust_label(pred_peaks,pred_labels,ratio)

        pred_peaks = pred_peaks*4
        
        Predict_peaks.append(pred_peaks)
        Predict_labels.append(pred_labels)
    
    if (true_peaks is not None) and (true_labels is not None) and (cl is not None):
        c = np.zeros((len(cl)+2,len(cl)+2))
        for i in range(len(Predict_peaks)):
            c += performance(Predict_peaks[i][1:-1],Predict_labels[i][1:-1],True_peaks[i][1:-1],True_labels[i][1:-1],cl)
    
        return X_signal, Predict_peaks, Predict_labels,True_peaks,True_labels,c
    
    else:
         return X_signal, Predict_peaks, Predict_labels

    





    
    
    
    
    

    
    
    


