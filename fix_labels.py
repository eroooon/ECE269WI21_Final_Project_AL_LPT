import numpy as np

def fix_labels(list_labels):
    t = np.array(list_labels[0])
    i = 1
    while i < list_labels.shape[0]:
        t2 = np.array(list_labels[i])
        t = np.vstack((t, t2))
        i = i + 1
    array_labels = t
    
    return array_labels