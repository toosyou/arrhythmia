import numpy as np
import matplotlib.pyplot as plt

def draw(signal, y_pred,true_peaks,true_labels,pred_peaks,pred_labels,num,name):
    
    color = ['bo','co','ro','go','ko']
    predict = y_pred.max(axis=-1)

    plt.figure(figsize = (20,10))
    plt.subplot(211)
    plt.plot(signal)s
    #true_peaks = true_peaks*4
    for i in range(true_labels.shape[0]):
        plt.plot(true_peaks[i],signal[true_peaks[i]],color[true_labels[i]])

    plt.subplot(212)
    plt.plot(predict)
    for i in range(pred_labels.shape[0]):
        plt.plot(pred_peaks[i],predict[pred_peaks[i]],color[pred_labels[i]])

    plt.show()
    plt.savefig("Image/"+str(num)+name)
    

