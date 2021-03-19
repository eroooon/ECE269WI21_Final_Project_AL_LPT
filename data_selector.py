import numpy as np

def data_selector(data):
    
    #   copy respective data parts    
    d = data[:,0]
    did = data[:,1]
    didx = data[:,2]
    

    #   Get stats of data
    du = np.mean(d)
    dstd = np.std(d, ddof=1)

    
    #   Test 1 (Medium)
    t1_val = d[np.abs(d - du) <= dstd]
    t1_idx = didx[np.abs(d - du) <= dstd]
    t1_class = did[np.abs(d - du) <= dstd]
    
    
    #   Test 2 (Hard)
    t2_val = d[(np.abs(d - du) > dstd) & (d < du)]
    t2_idx = didx[(np.abs(d - du) > dstd) & (d < du)]
    t2_class = did[(np.abs(d - du) > dstd) & (d < du)]
    

    #   Test 3 (Easy)
    t3_val = d[(np.abs(d - du) > dstd) & (d > du)]
    t3_idx = didx[(np.abs(d - du) > dstd) & (d > du)]
    t3_class = did[(np.abs(d - du) > dstd) & (d > du)]
            
    
    #   Concatenate for output
    easy = np.vstack([t3_val, t3_class, t3_idx])
    easy = np.transpose(easy, axes=None)
    medium = np.vstack([t1_val, t1_class, t1_idx])
    medium = np.transpose(medium, axes=None)
    hard = np.vstack([t2_val, t2_class, t2_idx])
    hard = np.transpose(hard, axes=None)
    
    return easy, medium, hard