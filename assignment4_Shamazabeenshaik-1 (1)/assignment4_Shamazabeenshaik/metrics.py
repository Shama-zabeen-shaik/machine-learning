def err(TtrainS, Tcal):
    import numpy as np
    e = 0
    for k in range(len(TtrainS)):  
        if np.sign(Tcal[k]) != np.sign(TtrainS[k]):
            e += 1
    e = e/len(TtrainS)
    return e

def metrics(TS, Tcal):
    import numpy as np
    TS = np.array(TS)
    Tcal = np.array(Tcal)
    
    TN = np.sum(np.logical_and(TS==-1,Tcal <0))
    FN = np.sum(np.logical_and(TS==1,Tcal <0))
    FP = np.sum(np.logical_and(TS==-1,Tcal >0))
    TP = np.sum(np.logical_and(TS==1,Tcal >0))  
    
    precision = TP/(TP+FP+0.01) *100
    recall = TP/(TP+FN+0.01) *100
    specificity = TN/(TN+FP+0.01) *100  # add 1 to avoid overflow 
#     F1 = TP/(TP + (FN+FP)/2)
#     MMC = (TP*TN-FP*FN)/(np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
    error = err(TS, Tcal)
    accuracy = (1-error) * 100
    
    return np.array([accuracy, recall, precision, specificity])
