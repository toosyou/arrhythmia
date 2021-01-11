import numpy as np
import pywt
from scipy.signal import medfilt

import multiprocessing as mp

from scipy.signal import butter, sosfiltfilt

def wavelet_threshold(data, wavelet='sym8', noiseSigma=10):
    levels = int(np.floor(np.log2(data.shape[0])))
    WC = pywt.wavedec(data,wavelet,level=levels)
    threshold=noiseSigma*np.sqrt(2*np.log2(data.size))
    NWC = list(map(lambda x: pywt.threshold(x,threshold, mode='soft'), WC))
    return pywt.waverec(NWC, wavelet)

def baseline_wander_removal(data, fs):
    def to_odd(x):
        x = int(x)
        return x if x % 2 == 1 else x + 1

    baseline = medfilt(data, to_odd(fs*0.2))
    baseline = medfilt(baseline, to_odd(fs*0.6))
    return data - baseline

def _denoise_mp(signal, fs):
    return baseline_wander_removal(wavelet_threshold(signal), fs)

def denoise(*args, **kwargs):
    import warnings
    warnings.warn('The denoise.denoise function is deprecated, use denoise.ekg_denoise instead!', UserWarning)
    return ekg_denoise(*args, **kwargs)

def ekg_denoise(data, fs, number_channels=None):
    '''Denoise the ekg data parallely and return.
    
    data: np.ndarray of shape [n_channels, n_samples]
    fs: sampling rate of data
    number_channels: the first N channels to be processed
    '''

    number_channels = data.shape[0] if number_channels is None else number_channels

    with mp.Pool(processes=number_channels) as workers:
        results = list()

        for i in range(number_channels):
            results.append(workers.apply_async(_denoise_mp, (data[i], fs)))

        workers.close()
        workers.join()

        for i, result in enumerate(results):
            data[i] = result.get()

    return data

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    '''Butter bandpass filter
    source: https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
    args:
        data: np.array of shape [n_samples]
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    y = sosfiltfilt(sos, data)
    return y

def heart_sound_denoise(data, lowcut, highcut, fs, order=5):
    '''Denoise heart sound signal with band pass filters and return.
    args:
        data: np.ndarray of shape [n_channels, n_samples]
    '''
    for index_channel in range(data.shape[0]):
        data[index_channel] = butter_bandpass_filter(data[index_channel], lowcut, highcut, fs, order)
    return data

