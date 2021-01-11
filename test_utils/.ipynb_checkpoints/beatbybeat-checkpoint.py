import numpy as np


def beatbybeat(fs, r_peaks, r_gt):
    
    #th = fs*0.15
    th = 100
    
    t_i = 0
    T_i = 0
    pair = []
    miss = []
    extra = []

    
    while T_i<len(r_gt) and t_i<len(r_peaks):
        t = r_peaks[t_i]
        T = r_gt[T_i]
        
        if t < T:
            if t_i+1 >= len(r_peaks):
                if abs(T - t) < th:
                    pair.append([T_i,t_i])
                else:
                    miss.append(T_i)
                    extra.append(t_i)   
                break
            else:
                t_p =  r_peaks[t_i+1]
                if abs(T - t) < abs(T - t_p) and abs(T - t) < th:
                    pair.append([T_i,t_i])
                    T_i += 1
                    t_i += 1
                else:
                    extra.append(t_i)
                    t_i += 1
        else:
            if T_i+1>=len(r_gt):
                if abs(t - T) < th:
                    pair.append([T_i,t_i])
                else:
                    miss.append(T_i)
                    extra.append(t_i)
                break
            else:
                T_p = r_gt[T_i+1]
                if abs(t - T) < abs(t - T_p) and abs(t - T) < th:
                    pair.append([T_i,t_i])
                    t_i += 1
                    T_i += 1
                else:
                    miss.append(T_i)
                    T_i += 1
                    
    if T_i != len(r_gt)-1:
        for i in range(T_i+1,len(r_gt),1):
            miss.append(T_i)
    if t_i != len(r_peaks)-1:
        for i in range(t_i+1,len(r_peaks),1):
            extra.append(t_i)
            
        
    return np.array(pair), np.array(miss), np.array(extra)