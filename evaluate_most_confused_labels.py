import numpy as np

def evaluate_most_confused_labels(conf_arr):
    # most_confused[tuple(truth, predicted)]
    copy_confusion_matrix = conf_arr
    most_cofused = []
    while len(most_cofused) != 2:
        max_index = np.where(copy_confusion_matrix == np.amax(copy_confusion_matrix))
        print(np.amax(copy_confusion_matrix))
        if (max_index[0] != max_index[1]):
            most_cofused.append(tuple((max_index[0], max_index[1])))
        copy_confusion_matrix[max_index[0],max_index[1]] = -1    
    print(most_cofused)
    
    return most_cofused